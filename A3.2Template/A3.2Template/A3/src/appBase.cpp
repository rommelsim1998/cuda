/*
* Vulkan Example - 
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
* Vulkan Application base class
*
* Modified by William Zheng
*
*/

#include "appBase.h"


std::vector<const char*> VkAppBase::args;
///@William
// Take a screenshot from the current swapchain image
// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependent optimal tiling format
// Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
void VkAppBase::saveScreenshot(const char* filename)
{
	screenshotSaved = false;
	bool supportsBlit = true;

	// Check blit support for source and destination
	VkFormatProperties formatProps;

	// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
	vkGetPhysicalDeviceFormatProperties(physicalDevice, swapChain.colorFormat, &formatProps);
	if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
		std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
		supportsBlit = false;
	}

	// Check if the device supports blitting to linear images
	vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
	if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
		std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
		supportsBlit = false;
	}

	// Source for the copy is the last rendered swapchain image
	VkImage srcImage = swapChain.images[currentBuffer];

	// Create the linear tiled destination image to copy to and to read the memory from
	VkImageCreateInfo imageCreateCI(vks::initializers::imageCreateInfo());
	imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
	// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
	imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageCreateCI.extent.width = width;
	imageCreateCI.extent.height = height;
	imageCreateCI.extent.depth = 1;
	imageCreateCI.arrayLayers = 1;
	imageCreateCI.mipLevels = 1;
	imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateCI.tiling = VK_IMAGE_TILING_LINEAR;
	imageCreateCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	// Create the image
	VkImage dstImage;
	VK_CHECK_RESULT(vkCreateImage(device, &imageCreateCI, nullptr, &dstImage));
	// Create memory to back up the image
	VkMemoryRequirements memRequirements;
	VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
	VkDeviceMemory dstImageMemory;
	vkGetImageMemoryRequirements(device, dstImage, &memRequirements);
	memAllocInfo.allocationSize = memRequirements.size;
	// Memory must be host visible to copy from
	memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &dstImageMemory));
	VK_CHECK_RESULT(vkBindImageMemory(device, dstImage, dstImageMemory, 0));

	// Do the actual blit from the swapchain image to our host visible destination image
	VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Transition destination image to transfer destination layout
	vks::tools::insertImageMemoryBarrier(
		copyCmd,
		dstImage,
		0,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	// Transition swapchain image from present to transfer source layout
	vks::tools::insertImageMemoryBarrier(
		copyCmd,
		srcImage,
		VK_ACCESS_MEMORY_READ_BIT,
		VK_ACCESS_TRANSFER_READ_BIT,
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
	if (supportsBlit)
	{
		// Define the region to blit (we will blit the whole swapchain image)
		VkOffset3D blitSize;
		blitSize.x = width;
		blitSize.y = height;
		blitSize.z = 1;
		VkImageBlit imageBlitRegion{};
		imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlitRegion.srcSubresource.layerCount = 1;
		imageBlitRegion.srcOffsets[1] = blitSize;
		imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlitRegion.dstSubresource.layerCount = 1;
		imageBlitRegion.dstOffsets[1] = blitSize;

		// Issue the blit command
		vkCmdBlitImage(
			copyCmd,
			srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageBlitRegion,
			VK_FILTER_NEAREST);
	}
	else
	{
		// Otherwise use image copy (requires us to manually flip components)
		VkImageCopy imageCopyRegion{};
		imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.srcSubresource.layerCount = 1;
		imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.dstSubresource.layerCount = 1;
		imageCopyRegion.extent.width = width;
		imageCopyRegion.extent.height = height;
		imageCopyRegion.extent.depth = 1;

		// Issue the copy command
		vkCmdCopyImage(
			copyCmd,
			srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageCopyRegion);
	}

	// Transition destination image to general layout, which is the required layout for mapping the image memory later on
	vks::tools::insertImageMemoryBarrier(
		copyCmd,
		dstImage,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_MEMORY_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	// Transition back the swap chain image after the blit is done
	vks::tools::insertImageMemoryBarrier(
		copyCmd,
		srcImage,
		VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_MEMORY_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	vulkanDevice->flushCommandBuffer(copyCmd, queue);

	// Get layout of the image (including row pitch)
	VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
	VkSubresourceLayout subResourceLayout;
	vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

	// Map image memory so we can start copying from it
	const char* data;
	vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)& data);
	data += subResourceLayout.offset;

	std::ofstream file(filename, std::ios::out | std::ios::binary);

	// ppm header
	file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

	// If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
	bool colorSwizzle = false;
	// Check if source is BGR
	// Note: Not complete, only contains most common and basic BGR surface formats for demonstration purposes
	if (!supportsBlit)
	{
		std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
		colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.colorFormat) != formatsBGR.end());
	}

	// ppm binary pixel data
	for (uint32_t y = 0; y < height; y++)
	{
		unsigned int* row = (unsigned int*)data;
		for (uint32_t x = 0; x < width; x++)
		{
			if (colorSwizzle)
			{
				file.write((char*)row + 2, 1);
				file.write((char*)row + 1, 1);
				file.write((char*)row, 1);
			}
			else
			{
				file.write((char*)row, 3);
			}
			row++;
		}
		data += subResourceLayout.rowPitch;
	}
	file.close();

	std::cout << "Screenshot saved to disk" << std::endl;

	// Clean up resources
	vkUnmapMemory(device, dstImageMemory);
	vkFreeMemory(device, dstImageMemory, nullptr);
	vkDestroyImage(device, dstImage, nullptr);

	if (!saveScreenshotOnce) {
		screenshotSaved = true;
	}
}

VkResult VkAppBase::createInstance(bool enableValidation)
{
	this->settings.validation = enableValidation;

	// Validation can also be forced via a define
#if defined(_VALIDATION)
	this->settings.validation = true;
#endif

	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = name.c_str();
	appInfo.pEngineName = name.c_str();
	appInfo.apiVersion = apiVersion;

	std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

	// Enable surface extensions depending on os
#if defined(_WIN32)
	instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif

	// Get extensions supported by the instance and store for later use
	uint32_t extCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
	if (extCount > 0)
	{
		std::vector<VkExtensionProperties> extensions(extCount);
		if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
		{
			for (VkExtensionProperties extension : extensions)
			{
				supportedInstanceExtensions.push_back(extension.extensionName);
			}
		}
	}

	// Enabled requested instance extensions
	if (enabledInstanceExtensions.size() > 0)
	{
		for (const char* enabledExtension : enabledInstanceExtensions)
		{
			// Output message if requested extension is not available
			if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
			{
				std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
			}
			instanceExtensions.push_back(enabledExtension);
		}
	}

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.pApplicationInfo = &appInfo;
	if (instanceExtensions.size() > 0)
	{
		if (settings.validation)
		{
			instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	}

	// The VK_LAYER_KHRONOS_validation contains all current validation functionality.
	const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
	if (settings.validation)
	{
		// Check if this layer is available at instance level
		uint32_t instanceLayerCount;
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
		std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
		bool validationLayerPresent = false;
		for (VkLayerProperties layer : instanceLayerProperties) {
			if (strcmp(layer.layerName, validationLayerName) == 0) {
				validationLayerPresent = true;
				break;
			}
		}
		if (validationLayerPresent) {
			instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
			instanceCreateInfo.enabledLayerCount = 1;
		}
		else {
			std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled";
		}
	}
	return vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
}

void VkAppBase::renderFrame()
{
	VkAppBase::prepareFrame();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VkAppBase::submitFrame();
}

std::string VkAppBase::getWindowTitle()
{
	std::string device(deviceProperties.deviceName);
	std::string windowTitle;
	windowTitle = title + " - " + device;
	if (!settings.overlay) {
		windowTitle += " - " + std::to_string(frameCounter) + " fps";
	}
	return windowTitle;
}

void VkAppBase::createCommandBuffers()
{
	// Create one command buffer for each swap chain image and reuse for rendering
	drawCmdBuffers.resize(swapChain.imageCount);

	VkCommandBufferAllocateInfo cmdBufAllocateInfo =
		vks::initializers::commandBufferAllocateInfo(
			cmdPool,
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			static_cast<uint32_t>(drawCmdBuffers.size()));

	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data()));
}

void VkAppBase::destroyCommandBuffers()
{
	vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());
}

std::string VkAppBase::getShadersPath() const
{
	return getAssetPath() + "shaders/" + shaderDir + "/";
}

void VkAppBase::createPipelineCache()
{
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

void VkAppBase::prepare()
{
	if (vulkanDevice->enableDebugMarkers) {
		vks::debugmarker::setup(device);
	}
	initSwapchain();
	createCommandPool();
	setupSwapChain();
	createCommandBuffers();
	createSynchronizationPrimitives();
	setupDepthStencil();
	setupRenderPass();
	createPipelineCache();
	setupFrameBuffer();
	settings.overlay = settings.overlay && (!benchmark.active);
	if (settings.overlay) {
		UIOverlay.device = vulkanDevice;
		UIOverlay.queue = queue;
		UIOverlay.shaders = {
			loadShader(getShadersPath() + "base/uioverlay.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "base/uioverlay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		UIOverlay.prepareResources();
		UIOverlay.preparePipeline(pipelineCache, renderPass);
	}
}

VkPipelineShaderStageCreateInfo VkAppBase::loadShader(std::string fileName, VkShaderStageFlagBits stage)
{
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = stage;

	shaderStage.module = vks::tools::loadShader(fileName.c_str(), device);

	shaderStage.pName = "main";
	assert(shaderStage.module != VK_NULL_HANDLE);
	shaderModules.push_back(shaderStage.module);
	return shaderStage;
}

void VkAppBase::nextFrame()
{
	auto tStart = std::chrono::high_resolution_clock::now();
	if (viewUpdated)
	{
		viewUpdated = false;
		viewChanged();
	}

	render();
	///@William
	///enable the following to generate reference
	if (screenshotSaved)
		saveScreenshot("reference_screenshot.ppm");

	frameCounter++;
	auto tEnd = std::chrono::high_resolution_clock::now();
	auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
	frameTimer = (float)tDiff / 1000.0f;
	camera.update(frameTimer);
	if (camera.moving())
	{
		viewUpdated = true;
	}
	// Convert to clamped timer value
	if (!paused)
	{
		timer += timerSpeed * frameTimer;
		if (timer > 1.0)
		{
			timer -= 1.0f;
		}
	}
	float fpsTimer = (float)(std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count());
	if (fpsTimer > 1000.0f)
	{
		lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));
#if defined(_WIN32)
		if (!settings.overlay) {
			std::string windowTitle = getWindowTitle();
			SetWindowText(window, windowTitle.c_str());
		}
#endif
		frameCounter = 0;
		lastTimestamp = tEnd;
	}
	// TODO: Cap UI overlay update rates
	updateOverlay();
}

void VkAppBase::renderLoop()
{
	if (benchmark.active) {
		benchmark.run([=] { render(); }, vulkanDevice->properties);
		vkDeviceWaitIdle(device);
		if (benchmark.filename != "") {
			benchmark.saveResults();
		}
		return;
	}

	destWidth = width;
	destHeight = height;
	lastTimestamp = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
	MSG msg;
	bool quitMessageReceived = false;
	while (!quitMessageReceived) {
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			if (msg.message == WM_QUIT) {
				quitMessageReceived = true;
				break;
			}
		}
		if (prepared && !IsIconic(window)) {
			nextFrame();
		}
	}
#endif
	// Flush device to make sure all resources can be freed
	if (device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(device);
	}
}

void VkAppBase::updateOverlay()
{
	if (!settings.overlay)
		return;

	ImGuiIO& io = ImGui::GetIO();

	io.DisplaySize = ImVec2((float)width, (float)height);
	io.DeltaTime = frameTimer;

	io.MousePos = ImVec2(mousePos.x, mousePos.y);
	io.MouseDown[0] = mouseButtons.left;
	io.MouseDown[1] = mouseButtons.right;

	ImGui::NewFrame();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
	ImGui::SetNextWindowPos(ImVec2(10, 10));
	ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("Vulkan App", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
	ImGui::TextUnformatted(title.c_str());
	ImGui::TextUnformatted(deviceProperties.deviceName);
	ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / lastFPS), lastFPS);

 
	ImGui::PushItemWidth(110.0f * UIOverlay.scale);
	OnUpdateUIOverlay(&UIOverlay);
	ImGui::PopItemWidth();


	ImGui::End();
	ImGui::PopStyleVar();
	ImGui::Render();

	if (UIOverlay.update() || UIOverlay.updated) {
		buildCommandBuffers();
		UIOverlay.updated = false;
	}

 
}

void VkAppBase::drawUI(const VkCommandBuffer commandBuffer)
{
	if (settings.overlay) {
		const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		UIOverlay.draw(commandBuffer);
	}
}

void VkAppBase::prepareFrame()
{
	// Acquire the next image from the swap chain
	VkResult result = swapChain.acquireNextImage(semaphores.presentComplete, &currentBuffer);
	// Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
	if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
		windowResize();
	}
	else {
		VK_CHECK_RESULT(result);
	}
}

void VkAppBase::submitFrame()
{
	VkResult result = swapChain.queuePresent(queue, currentBuffer, semaphores.renderComplete);
	if (!((result == VK_SUCCESS) || (result == VK_SUBOPTIMAL_KHR))) {
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			// Swap chain is no longer compatible with the surface and needs to be recreated
			windowResize();
			return;
		}
		else {
			VK_CHECK_RESULT(result);
		}
	}
	VK_CHECK_RESULT(vkQueueWaitIdle(queue));
}

VkAppBase::VkAppBase(bool enableValidation)
{
	// Check for a valid asset path
	struct stat info;
	if (stat(getAssetPath().c_str(), &info) != 0)
	{
#if defined(_WIN32)
		std::string msg = "Could not locate asset path in \"" + getAssetPath() + "\" !";
		MessageBox(NULL, msg.c_str(), "Fatal error", MB_OK | MB_ICONERROR);
#else
		std::cerr << "Error: Could not find asset path in " << getAssetPath() << "\n";
#endif
		exit(-1);
	}

	settings.validation = enableValidation;

	// Command line arguments
	commandLineParser.parse(args);
	if (commandLineParser.isSet("help")) {
#if defined(_WIN32)
		setupConsole("Vulkan App");
#endif
		commandLineParser.printHelp();
		std::cin.get();
		exit(0);
	}
	if (commandLineParser.isSet("validation")) {
		settings.validation = true;
	}
	if (commandLineParser.isSet("vsync")) {
		settings.vsync = true;
	}
	if (commandLineParser.isSet("height")) {
		height = commandLineParser.getValueAsInt("height", width);
	}
	if (commandLineParser.isSet("width")) {
		width = commandLineParser.getValueAsInt("width", width);
	}
	if (commandLineParser.isSet("fullscreen")) {
		settings.fullscreen = true;
	}
	if (commandLineParser.isSet("shaders")) {
		std::string value = commandLineParser.getValueAsString("shaders", "glsl");
		if ((value != "glsl") && (value != "hlsl")) {
			std::cerr << "Shader type must be one of 'glsl' or 'hlsl'\n";
		}
		else {
			shaderDir = value;
		}
	}
	if (commandLineParser.isSet("benchmark")) {
		benchmark.active = true;
		vks::tools::errorModeSilent = true;
	}
	if (commandLineParser.isSet("benchmarkwarmup")) {
		benchmark.warmup = commandLineParser.getValueAsInt("benchmarkwarmup", benchmark.warmup);
	}
	if (commandLineParser.isSet("benchmarkruntime")) {
		benchmark.duration = commandLineParser.getValueAsInt("benchmarkruntime", benchmark.duration);
	}
	if (commandLineParser.isSet("benchmarkresultfile")) {
		benchmark.filename = commandLineParser.getValueAsString("benchmarkresultfile", benchmark.filename);
	}
	if (commandLineParser.isSet("benchmarkresultframes")) {
		benchmark.outputFrameTimes = true;
	}
	if (commandLineParser.isSet("benchmarkframes")) {
		benchmark.outputFrames = commandLineParser.getValueAsInt("benchmarkframes", benchmark.outputFrames);
	}
	///@William
	if (commandLineParser.isSet("sourcefile")) {
		benchmark.sourcefile = commandLineParser.getValueAsString("sourcefile", benchmark.sourcefile);
	}

#if defined(_WIN32)
	// Enable console if validation is active, debug message callback will output to it
	if (this->settings.validation)
	{
		setupConsole("Vulkan App");
	}
	setupDPIAwareness();
#endif
}

VkAppBase::~VkAppBase()
{
	// Clean up Vulkan resources
	swapChain.cleanup();
	if (descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	}
	destroyCommandBuffers();
	if (renderPass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(device, renderPass, nullptr);
	}
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
	}

	for (auto& shaderModule : shaderModules)
	{
		vkDestroyShaderModule(device, shaderModule, nullptr);
	}
	vkDestroyImageView(device, depthStencil.view, nullptr);
	vkDestroyImage(device, depthStencil.image, nullptr);
	vkFreeMemory(device, depthStencil.mem, nullptr);

	vkDestroyPipelineCache(device, pipelineCache, nullptr);

	vkDestroyCommandPool(device, cmdPool, nullptr);

	vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
	vkDestroySemaphore(device, semaphores.renderComplete, nullptr);
	for (auto& fence : waitFences) {
		vkDestroyFence(device, fence, nullptr);
	}

	if (settings.overlay) {
		UIOverlay.freeResources();
	}

	delete vulkanDevice;

	if (settings.validation)
	{
		vks::debug::freeDebugCallback(instance);
	}

	vkDestroyInstance(instance, nullptr);
}

bool VkAppBase::initVulkan()
{
	VkResult err;

	// Vulkan instance
	err = createInstance(settings.validation);
	if (err) {
		vks::tools::exitFatal("Could not create Vulkan instance : \n" + vks::tools::errorString(err), err);
		return false;
	}


	// If requested, we enable the default validation layers for debugging
	if (settings.validation)
	{
		// The report flags determine what type of messages for the layers will be displayed
		// For validating (debugging) an application the error and warning bits should suffice
		VkDebugReportFlagsEXT debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		// Additional flags include performance info, loader and layer debug messages, etc.
		vks::debug::setupDebugging(instance, debugReportFlags, VK_NULL_HANDLE);
	}

	// Physical device
	uint32_t gpuCount = 0;
	// Get number of available physical devices
	VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
	if (gpuCount == 0) {
		vks::tools::exitFatal("No device with Vulkan support found", -1);
		return false;
	}
	// Enumerate devices
	std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
	err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
	if (err) {
		vks::tools::exitFatal("Could not enumerate physical devices : \n" + vks::tools::errorString(err), err);
		return false;
	}

	// GPU selection

	// Select physical device to be used for the Vulkan example
	// Defaults to the first device unless specified by command line
	uint32_t selectedDevice = 0;

 
	// GPU selection via command line argument
	if (commandLineParser.isSet("gpuselection")) {
		uint32_t index = commandLineParser.getValueAsInt("gpuselection", 0);
		if (index > gpuCount - 1) {
			std::cerr << "Selected device index " << index << " is out of range, reverting to device 0 (use -listgpus to show available Vulkan devices)" << "\n";
		}
		else {
			selectedDevice = index;
		}
	}
	if (commandLineParser.isSet("gpulist")) {
		std::cout << "Available Vulkan devices" << "\n";
		for (uint32_t i = 0; i < gpuCount; i++) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevices[i], &deviceProperties);
			std::cout << "Device [" << i << "] : " << deviceProperties.deviceName << std::endl;
			std::cout << " Type: " << vks::tools::physicalDeviceTypeString(deviceProperties.deviceType) << "\n";
			std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "." << ((deviceProperties.apiVersion >> 12) & 0x3ff) << "." << (deviceProperties.apiVersion & 0xfff) << "\n";
		}
	}
 

	physicalDevice = physicalDevices[selectedDevice];

	// Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

	// Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
	getEnabledFeatures();

	// Vulkan device creation
	// This is handled by a separate class that gets a logical device representation
	// and encapsulates functions related to a device
	vulkanDevice = new vks::VulkanDevice(physicalDevice);
	VkResult res = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
	if (res != VK_SUCCESS) {
		vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(res), res);
		return false;
	}
	device = vulkanDevice->logicalDevice;

	// Get a graphics queue from the device
	vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);

	// Find a suitable depth format
	VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat);
	assert(validDepthFormat);

	swapChain.connect(instance, physicalDevice, device);

	// Create synchronization objects
	VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
	// Create a semaphore used to synchronize image presentation
	// Ensures that the image is displayed before we start submitting new commands to the queue
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete));
	// Create a semaphore used to synchronize command submission
	// Ensures that the image is not presented until all commands have been submitted and executed
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.renderComplete));

	// Set up submit info structure
	// Semaphores will stay the same during application lifetime
	// Command buffer submission info is set by each example
	submitInfo = vks::initializers::submitInfo();
	submitInfo.pWaitDstStageMask = &submitPipelineStages;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &semaphores.presentComplete;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &semaphores.renderComplete;

	return true;
}

#if defined(_WIN32)
// Win32 : Sets up a console window and redirects standard output to it
void VkAppBase::setupConsole(std::string title)
{
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	FILE* stream;
	freopen_s(&stream, "CONIN$", "r", stdin);
	freopen_s(&stream, "CONOUT$", "w+", stdout);
	freopen_s(&stream, "CONOUT$", "w+", stderr);
	SetConsoleTitle(TEXT(title.c_str()));
}

void VkAppBase::setupDPIAwareness()
{
	typedef HRESULT* (__stdcall* SetProcessDpiAwarenessFunc)(PROCESS_DPI_AWARENESS);

	HMODULE shCore = LoadLibraryA("Shcore.dll");
	if (shCore)
	{
		SetProcessDpiAwarenessFunc setProcessDpiAwareness =
			(SetProcessDpiAwarenessFunc)GetProcAddress(shCore, "SetProcessDpiAwareness");

		if (setProcessDpiAwareness != nullptr)
		{
			setProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
		}

		FreeLibrary(shCore);
	}
}

HWND VkAppBase::setupWindow(HINSTANCE hinstance, WNDPROC wndproc)
{
	this->windowInstance = hinstance;

	WNDCLASSEX wndClass;

	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_HREDRAW | CS_VREDRAW;
	wndClass.lpfnWndProc = wndproc;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hinstance;
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndClass.lpszMenuName = NULL;
	wndClass.lpszClassName = name.c_str();
	wndClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);

	if (!RegisterClassEx(&wndClass))
	{
		std::cout << "Could not register window class!\n";
		fflush(stdout);
		exit(1);
	}

	int screenWidth = GetSystemMetrics(SM_CXSCREEN);
	int screenHeight = GetSystemMetrics(SM_CYSCREEN);

	if (settings.fullscreen)
	{
		if ((width != (uint32_t)screenWidth) && (height != (uint32_t)screenHeight))
		{
			DEVMODE dmScreenSettings;
			memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
			dmScreenSettings.dmSize = sizeof(dmScreenSettings);
			dmScreenSettings.dmPelsWidth = width;
			dmScreenSettings.dmPelsHeight = height;
			dmScreenSettings.dmBitsPerPel = 32;
			dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
			if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
			{
				if (MessageBox(NULL, "Fullscreen Mode not supported!\n Switch to window mode?", "Error", MB_YESNO | MB_ICONEXCLAMATION) == IDYES)
				{
					settings.fullscreen = false;
				}
				else
				{
					return nullptr;
				}
			}
			screenWidth = width;
			screenHeight = height;
		}

	}

	DWORD dwExStyle;
	DWORD dwStyle;

	if (settings.fullscreen)
	{
		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
	}

	RECT windowRect;
	windowRect.left = 0L;
	windowRect.top = 0L;
	windowRect.right = settings.fullscreen ? (long)screenWidth : (long)width;
	windowRect.bottom = settings.fullscreen ? (long)screenHeight : (long)height;

	AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);

	std::string windowTitle = getWindowTitle();
	window = CreateWindowEx(0,
		name.c_str(),
		windowTitle.c_str(),
		dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		0,
		0,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		NULL,
		NULL,
		hinstance,
		NULL);

	if (!settings.fullscreen)
	{
		// Center on screen
		uint32_t x = (GetSystemMetrics(SM_CXSCREEN) - windowRect.right) / 2;
		uint32_t y = (GetSystemMetrics(SM_CYSCREEN) - windowRect.bottom) / 2;
		SetWindowPos(window, 0, x, y, 0, 0, SWP_NOZORDER | SWP_NOSIZE);
	}

	if (!window)
	{
		printf("Could not create window!\n");
		fflush(stdout);
		return nullptr;
	}

	ShowWindow(window, SW_SHOW);
	SetForegroundWindow(window);
	SetFocus(window);

	return window;
}

void VkAppBase::handleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_CLOSE:
		prepared = false;
		DestroyWindow(hWnd);
		PostQuitMessage(0);
		break;
	case WM_PAINT:
		ValidateRect(window, NULL);
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case KEY_P:
			paused = !paused;
			break;
		case KEY_F1:
			if (settings.overlay) {
				UIOverlay.visible = !UIOverlay.visible;
			}
			break;
		case KEY_ESCAPE:
			PostQuitMessage(0);
			break;
		}

		if (camera.type == Camera::firstperson)
		{
			switch (wParam)
			{
			case KEY_W:
				camera.keys.up = true;
				break;
			case KEY_S:
				camera.keys.down = true;
				break;
			case KEY_A:
				camera.keys.left = true;
				break;
			case KEY_D:
				camera.keys.right = true;
				break;
			}
		}

		keyPressed((uint32_t)wParam);
		break;
	case WM_KEYUP:
		if (camera.type == Camera::firstperson)
		{
			switch (wParam)
			{
			case KEY_W:
				camera.keys.up = false;
				break;
			case KEY_S:
				camera.keys.down = false;
				break;
			case KEY_A:
				camera.keys.left = false;
				break;
			case KEY_D:
				camera.keys.right = false;
				break;
			}
		}
		break;
	case WM_LBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.left = true;
		break;
	case WM_RBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.right = true;
		break;
	case WM_MBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.middle = true;
		break;
	case WM_LBUTTONUP:
		mouseButtons.left = false;
		break;
	case WM_RBUTTONUP:
		mouseButtons.right = false;
		break;
	case WM_MBUTTONUP:
		mouseButtons.middle = false;
		break;
	case WM_MOUSEWHEEL:
	{
		short wheelDelta = GET_WHEEL_DELTA_WPARAM(wParam);
		camera.translate(glm::vec3(0.0f, 0.0f, (float)wheelDelta * 0.005f));
		viewUpdated = true;
		break;
	}
	case WM_MOUSEMOVE:
	{
		handleMouseMove(LOWORD(lParam), HIWORD(lParam));
		break;
	}
	case WM_SIZE:
		if ((prepared) && (wParam != SIZE_MINIMIZED))
		{
			if ((resizing) || ((wParam == SIZE_MAXIMIZED) || (wParam == SIZE_RESTORED)))
			{
				destWidth = LOWORD(lParam);
				destHeight = HIWORD(lParam);
				windowResize();
			}
		}
		break;
	case WM_GETMINMAXINFO:
	{
		LPMINMAXINFO minMaxInfo = (LPMINMAXINFO)lParam;
		minMaxInfo->ptMinTrackSize.x = 64;
		minMaxInfo->ptMinTrackSize.y = 64;
		break;
	}
	case WM_ENTERSIZEMOVE:
		resizing = true;
		break;
	case WM_EXITSIZEMOVE:
		resizing = false;
		break;
	}
}
 
#else
void VkAppBase::setupWindow()
{
}
#endif

void VkAppBase::viewChanged() {}

void VkAppBase::keyPressed(uint32_t) {}

void VkAppBase::mouseMoved(double x, double y, bool& handled) {}

void VkAppBase::buildCommandBuffers() {}

void VkAppBase::createSynchronizationPrimitives()
{
	// Wait fences to sync command buffer access
	VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
	waitFences.resize(drawCmdBuffers.size());
	for (auto& fence : waitFences) {
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
	}
}

void VkAppBase::createCommandPool()
{
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VkAppBase::setupDepthStencil()
{
	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = depthFormat;
	imageCI.extent = { width, height, 1 };
	imageCI.mipLevels = 1;
	imageCI.arrayLayers = 1;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));
	VkMemoryRequirements memReqs{};
	vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

	VkMemoryAllocateInfo memAllloc{};
	memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllloc.allocationSize = memReqs.size;
	memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &depthStencil.mem));
	VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

	VkImageViewCreateInfo imageViewCI{};
	imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCI.image = depthStencil.image;
	imageViewCI.format = depthFormat;
	imageViewCI.subresourceRange.baseMipLevel = 0;
	imageViewCI.subresourceRange.levelCount = 1;
	imageViewCI.subresourceRange.baseArrayLayer = 0;
	imageViewCI.subresourceRange.layerCount = 1;
	imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
	if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
		imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &depthStencil.view));
}

void VkAppBase::setupFrameBuffer()
{
	VkImageView attachments[2];

	// Depth/Stencil attachment is the same for all frame buffers
	attachments[1] = depthStencil.view;

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext = NULL;
	frameBufferCreateInfo.renderPass = renderPass;
	frameBufferCreateInfo.attachmentCount = 2;
	frameBufferCreateInfo.pAttachments = attachments;
	frameBufferCreateInfo.width = width;
	frameBufferCreateInfo.height = height;
	frameBufferCreateInfo.layers = 1;

	// Create frame buffers for every swap chain image
	frameBuffers.resize(swapChain.imageCount);
	//std::cout << "frameBuffers size =  " << frameBuffers.size() << std::endl;
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		attachments[0] = swapChain.buffers[i].view;
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
	}
}

void VkAppBase::setupRenderPass()
{
	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment
	attachments[0].format = swapChain.colorFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	// Depth attachment
	attachments[1].format = depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = nullptr;
	subpassDescription.pResolveAttachments = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies = dependencies.data();

	VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void VkAppBase::getEnabledFeatures() {}

void VkAppBase::windowResize()
{
	if (!prepared)
	{
		return;
	}
	prepared = false;
	resized = true;

	// Ensure all operations on the device have been finished before destroying resources
	vkDeviceWaitIdle(device);

	// Recreate swap chain
	width = destWidth;
	height = destHeight;
	setupSwapChain();

	// Recreate the frame buffers
	vkDestroyImageView(device, depthStencil.view, nullptr);
	vkDestroyImage(device, depthStencil.image, nullptr);
	vkFreeMemory(device, depthStencil.mem, nullptr);
	setupDepthStencil();
	for (uint32_t i = 0; i < frameBuffers.size(); i++) {
		vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
	}
	setupFrameBuffer();

	if ((width > 0.0f) && (height > 0.0f)) {
		if (settings.overlay) {
			UIOverlay.resize(width, height);
		}
	}

	// Command buffers need to be recreated as they may store
	// references to the recreated frame buffer
	destroyCommandBuffers();
	createCommandBuffers();
	buildCommandBuffers();

	vkDeviceWaitIdle(device);

	if ((width > 0.0f) && (height > 0.0f)) {
		camera.updateAspectRatio((float)width / (float)height);
	}

	// Notify derived class
	windowResized();
	viewChanged();

	prepared = true;
}

void VkAppBase::handleMouseMove(int32_t x, int32_t y)
{
	int32_t dx = (int32_t)mousePos.x - x;
	int32_t dy = (int32_t)mousePos.y - y;

	bool handled = false;

	if (settings.overlay) {
		ImGuiIO& io = ImGui::GetIO();
		handled = io.WantCaptureMouse;
	}
	mouseMoved((float)x, (float)y, handled);

	if (handled) {
		mousePos = glm::vec2((float)x, (float)y);
		return;
	}

	if (mouseButtons.left) {
		camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
		viewUpdated = true;
	}
	if (mouseButtons.right) {
		camera.translate(glm::vec3(-0.0f, 0.0f, dy * .005f));
		viewUpdated = true;
	}
	if (mouseButtons.middle) {
		camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
		viewUpdated = true;
	}
	mousePos = glm::vec2((float)x, (float)y);
}

void VkAppBase::windowResized() {}

void VkAppBase::initSwapchain()
{
#if defined(_WIN32)
	swapChain.initSurface(windowInstance, window);
#endif
}

void VkAppBase::setupSwapChain()
{
	swapChain.create(&width, &height, settings.vsync);
}

void VkAppBase::OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

// Command line argument parser class

CommandLineParser::CommandLineParser()
{
	add("help", { "--help" }, 0, "Show help");
	add("validation", { "-v", "--validation" }, 0, "Enable validation layers");
	add("vsync", { "-vs", "--vsync" }, 0, "Enable V-Sync");
	add("fullscreen", { "-f", "--fullscreen" }, 0, "Start in fullscreen mode");
	add("width", { "-w", "--width" }, 1, "Set window width");
	add("height", { "-h", "--height" }, 1, "Set window height");
	add("shaders", { "-s", "--shaders" }, 1, "Select shader type to use (glsl or hlsl)");
	add("gpuselection", { "-g", "--gpu" }, 1, "Select GPU to run on");
	add("gpulist", { "-gl", "--listgpus" }, 0, "Display a list of available Vulkan devices");
	add("benchmark", { "-b", "--benchmark" }, 0, "Run example in benchmark mode");
	add("benchmarkwarmup", { "-bw", "--benchwarmup" }, 1, "Set warmup time for benchmark mode in seconds");
	add("benchmarkruntime", { "-br", "--benchruntime" }, 1, "Set duration time for benchmark mode in seconds");
	add("benchmarkresultfile", { "-bf", "--benchfilename" }, 1, "Set file name for benchmark results");
	add("benchmarkresultframes", { "-bt", "--benchframetimes" }, 0, "Save frame times to benchmark results file");
	add("benchmarkframes", { "-bfs", "--benchmarkframes" }, 1, "Only render the given number of frames");
	//@@@william
	add("sourcefile", { "-sf", "--sourcefile" }, 1, "load ktx format source file for image processing");
}

void CommandLineParser::add(std::string name, std::vector<std::string> commands, bool hasValue, std::string help)
{
	options[name].commands = commands;
	options[name].help = help;
	options[name].set = false;
	options[name].hasValue = hasValue;
	options[name].value = "";
}

void CommandLineParser::printHelp()
{
	std::cout << "Available command line options:\n";
	for (auto option : options) {
		std::cout << " ";
		for (size_t i = 0; i < option.second.commands.size(); i++) {
			std::cout << option.second.commands[i];
			if (i < option.second.commands.size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << ": " << option.second.help << "\n";
	}
	std::cout << "Press any key to close...";
}

void CommandLineParser::parse(std::vector<const char*> arguments)
{
	bool printHelp = false;
	// Known arguments
	for (auto& option : options) {
		for (auto& command : option.second.commands) {
			for (size_t i = 0; i < arguments.size(); i++) {
				if (strcmp(arguments[i], command.c_str()) == 0) {
					option.second.set = true;
					// Get value
					if (option.second.hasValue) {
						if (arguments.size() > i + 1) {
							option.second.value = arguments[i + 1];
						}
						if (option.second.value == "") {
							printHelp = true;
							break;
						}
					}
				}
			}
		}
	}
	// Print help for unknown arguments or missing argument values
	if (printHelp) {
		options["help"].set = true;
	}
}

bool CommandLineParser::isSet(std::string name)
{
	return ((options.find(name) != options.end()) && options[name].set);
}

std::string CommandLineParser::getValueAsString(std::string name, std::string defaultValue)
{
	assert(options.find(name) != options.end());
	std::string value = options[name].value;
	return (value != "") ? value : defaultValue;
}

int32_t CommandLineParser::getValueAsInt(std::string name, int32_t defaultValue)
{
	assert(options.find(name) != options.end());
	std::string value = options[name].value;
	if (value != "") {
		char* numConvPtr;
		int32_t intVal = strtol(value.c_str(), &numConvPtr, 10);
		return (intVal > 0) ? intVal : defaultValue;
	}
	else {
		return defaultValue;
	}
	return int32_t();
}
