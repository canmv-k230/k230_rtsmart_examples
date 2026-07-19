#include "xiaozhi_mcp_internal.h"

#include <cJSON.h>

static cJSON *make_light_state(int on)
{
	cJSON *state = cJSON_CreateObject();

	if (!state)
		return NULL;
	cJSON_AddBoolToObject(state, "available", 1);
	cJSON_AddBoolToObject(state, "on", on != 0);
	return state;
}

static cJSON *make_light_result(int on)
{
	cJSON *state;
	cJSON *result;

	state = make_light_state(on);
	if (!state)
		return NULL;
	result = xiaozhi_mcp_result_json(state);
	cJSON_Delete(state);
	return result;
}

static int tool_get_light(void *opaque, const cJSON *arguments,
				  cJSON **result, char *error, size_t error_size)
{
	const struct xiaozhi_mcp_device *device =
		(const struct xiaozhi_mcp_device *)opaque;
	int on;

	(void)arguments;
	if (!device || !device->light_available || !device->get_light) {
		xiaozhi_mcp_set_error(error, error_size,
				      "GPIO light state is unavailable");
		return -1;
	}
	if (device->get_light(device->opaque, &on)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to get GPIO light state");
		return -1;
	}
	*result = make_light_result(on);
	if (!*result) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to build tool result");
		return -1;
	}
	return 0;
}

static int tool_set_light(void *opaque, const cJSON *arguments,
				  cJSON **result, char *error, size_t error_size)
{
	const struct xiaozhi_mcp_device *device =
		(const struct xiaozhi_mcp_device *)opaque;
	const cJSON *state;

	if (!device || !device->light_available || !device->set_light ||
	    !cJSON_IsObject(arguments)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "GPIO light is unavailable");
		return -1;
	}
	state = cJSON_GetObjectItemCaseSensitive(arguments, "state");
	if (!cJSON_IsBool(state)) {
		xiaozhi_mcp_set_error(error, error_size,
				      "state must be a boolean");
		return -1;
	}
	if (device->set_light(device->opaque, cJSON_IsTrue(state))) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to set GPIO light");
		return -1;
	}
	*result = make_light_result(cJSON_IsTrue(state));
	if (!*result) {
		xiaozhi_mcp_set_error(error, error_size,
				      "Failed to build tool result");
		return -1;
	}
	return 0;
}

static const struct xiaozhi_mcp_property light_properties[] = {
	{
		.name = "state",
		.description = "Whether the light is on.",
		.type = XIAOZHI_MCP_PROPERTY_BOOLEAN,
		.required = 1,
	},
};

static const struct xiaozhi_mcp_tool light_get_tool = {
	.name = "self.light.get_state",
	.description = "Get the light state. The result contains available=true and on=true or false; on=false means the light is configured and currently off.",
	.call = tool_get_light,
};

static const struct xiaozhi_mcp_tool light_set_tool = {
	.name = "self.light.set_state",
	.description = "Turn the light on or off. The result contains available=true and the resulting on state.",
	.properties = light_properties,
	.property_count = sizeof(light_properties) / sizeof(light_properties[0]),
	.call = tool_set_light,
};

const struct xiaozhi_mcp_tool *xiaozhi_mcp_light_get_tool(void)
{
	return &light_get_tool;
}

const struct xiaozhi_mcp_tool *xiaozhi_mcp_light_set_tool(void)
{
	return &light_set_tool;
}
