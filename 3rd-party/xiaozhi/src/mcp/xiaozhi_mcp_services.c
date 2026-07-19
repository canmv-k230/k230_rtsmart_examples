#include "xiaozhi_mcp_internal.h"

int xiaozhi_mcp_register_default_tools(struct xiaozhi_mcp_server *server)
{
	const struct xiaozhi_mcp_device *device;

	if (!server)
		return -1;
	device = server->device;
	if (device && device->get_volume &&
	    xiaozhi_mcp_register_tool(server,
				       xiaozhi_mcp_audio_get_volume_tool()))
		return -1;
	if (device && device->set_volume &&
	    xiaozhi_mcp_register_tool(server,
				       xiaozhi_mcp_audio_set_volume_tool()))
		return -1;
	if (device && device->light_available && device->get_light &&
	    xiaozhi_mcp_register_tool(server, xiaozhi_mcp_light_get_tool()))
		return -1;
	if (device && device->light_available && device->set_light &&
	    xiaozhi_mcp_register_tool(server, xiaozhi_mcp_light_set_tool()))
		return -1;
	return 0;
}
