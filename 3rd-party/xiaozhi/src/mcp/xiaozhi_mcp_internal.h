#ifndef XIAOZHI_MCP_INTERNAL_H
#define XIAOZHI_MCP_INTERNAL_H

#include "xiaozhi_mcp.h"

#include <stddef.h>

void xiaozhi_mcp_set_error(char *error, size_t error_size,
				   const char *message);

const struct xiaozhi_mcp_tool *xiaozhi_mcp_audio_get_volume_tool(void);
const struct xiaozhi_mcp_tool *xiaozhi_mcp_audio_set_volume_tool(void);
const struct xiaozhi_mcp_tool *xiaozhi_mcp_light_get_tool(void);
const struct xiaozhi_mcp_tool *xiaozhi_mcp_light_set_tool(void);

#endif
