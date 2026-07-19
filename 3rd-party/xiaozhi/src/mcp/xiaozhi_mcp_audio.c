#include "xiaozhi_mcp_internal.h"

#include <cJSON.h>
#include <stdio.h>

static int tool_get_volume(void *opaque, const cJSON *arguments,
				   cJSON **result, char *error, size_t error_size)
{
	const struct xiaozhi_mcp_device *device =
		(const struct xiaozhi_mcp_device *)opaque;
	char volume_text[16];
	int volume;

	(void)arguments;
	if (!device || !device->get_volume) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Speaker volume is unavailable");
		return -1;
	}
	if (device->get_volume(device->opaque, &volume)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to get speaker volume");
		return -1;
	}
	if (snprintf(volume_text, sizeof(volume_text), "%d", volume) < 0) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to format speaker volume");
		return -1;
	}
	*result = xiaozhi_mcp_result_text(volume_text);
	if (!*result) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to build tool result");
		return -1;
	}
	return 0;
}

static int tool_set_volume(void *opaque, const cJSON *arguments,
				   cJSON **result, char *error, size_t error_size)
{
	const struct xiaozhi_mcp_device *device =
		(const struct xiaozhi_mcp_device *)opaque;
	const cJSON *volume;

	if (!device || !device->set_volume || !cJSON_IsObject(arguments)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Speaker volume is unavailable");
		return -1;
	}
	volume = cJSON_GetObjectItemCaseSensitive(arguments, "volume");
	if (!cJSON_IsNumber(volume) || volume->valuedouble != volume->valueint ||
	    volume->valueint < 0 || volume->valueint > 100) {
		xiaozhi_mcp_set_error(error, error_size,
				      "volume must be an integer from 0 to 100");
		return -1;
	}
	if (device->set_volume(device->opaque, volume->valueint)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to set speaker volume");
		return -1;
	}
	*result = xiaozhi_mcp_result_bool(1);
	if (!*result) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to build tool result");
		return -1;
	}
	return 0;
}

static const struct xiaozhi_mcp_property volume_properties[] = {
	{
		.name = "volume",
		.description = "Speaker volume from 0 to 100.",
		.type = XIAOZHI_MCP_PROPERTY_INTEGER,
		.required = 1,
		.has_minimum = 1,
		.has_maximum = 1,
		.minimum = 0,
		.maximum = 100,
	},
};

static const struct xiaozhi_mcp_tool audio_get_volume_tool = {
	.name = "self.audio_speaker.get_volume",
	.description = "Get the current audio speaker volume from 0 to 100.",
	.call = tool_get_volume,
};

static const struct xiaozhi_mcp_tool audio_set_volume_tool = {
	.name = "self.audio_speaker.set_volume",
	.description = "Set the volume of the audio speaker. Call self.audio_speaker.get_volume first when the current volume is unknown.",
	.properties = volume_properties,
	.property_count = sizeof(volume_properties) / sizeof(volume_properties[0]),
	.call = tool_set_volume,
};

const struct xiaozhi_mcp_tool *xiaozhi_mcp_audio_get_volume_tool(void)
{
	return &audio_get_volume_tool;
}

const struct xiaozhi_mcp_tool *xiaozhi_mcp_audio_set_volume_tool(void)
{
	return &audio_set_volume_tool;
}
