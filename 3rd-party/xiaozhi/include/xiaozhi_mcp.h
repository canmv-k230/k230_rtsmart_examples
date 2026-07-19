#ifndef XIAOZHI_MCP_H
#define XIAOZHI_MCP_H

#include <cJSON.h>

#include <stddef.h>

#define XIAOZHI_MCP_MAX_TOOLS 64

enum xiaozhi_mcp_property_type {
	XIAOZHI_MCP_PROPERTY_BOOLEAN,
	XIAOZHI_MCP_PROPERTY_INTEGER,
	XIAOZHI_MCP_PROPERTY_NUMBER,
	XIAOZHI_MCP_PROPERTY_STRING,
	XIAOZHI_MCP_PROPERTY_OBJECT,
	XIAOZHI_MCP_PROPERTY_ARRAY,
};

struct xiaozhi_mcp_property {
	const char *name;
	const char *description;
	enum xiaozhi_mcp_property_type type;
	int required;
	int has_minimum;
	int has_maximum;
	double minimum;
	double maximum;
};

struct xiaozhi_mcp_device {
	const char *name;
	const char *version;
	int (*set_volume)(void *opaque, int volume);
	int (*get_volume)(void *opaque, int *volume);
	int light_available;
	int (*set_light)(void *opaque, int on);
	int (*get_light)(void *opaque, int *on);
	void *opaque;
};

struct xiaozhi_mcp_tool;

typedef int (*xiaozhi_mcp_tool_call_fn)(
	void *opaque, const cJSON *arguments, cJSON **result,
	char *error, size_t error_size);

struct xiaozhi_mcp_tool {
	const char *name;
	const char *description;
	const struct xiaozhi_mcp_property *properties;
	size_t property_count;
	int user_only;
	xiaozhi_mcp_tool_call_fn call;
	void *opaque;
};

struct xiaozhi_mcp_server {
	const struct xiaozhi_mcp_device *device;
	const struct xiaozhi_mcp_tool *tools[XIAOZHI_MCP_MAX_TOOLS];
	size_t tool_count;
};

typedef int (*xiaozhi_mcp_send_fn)(void *opaque, const char *payload,
					   size_t len);

int xiaozhi_mcp_server_init(struct xiaozhi_mcp_server *server,
				    const struct xiaozhi_mcp_device *device);

int xiaozhi_mcp_register_tool(struct xiaozhi_mcp_server *server,
				      const struct xiaozhi_mcp_tool *tool);

int xiaozhi_mcp_register_tools(struct xiaozhi_mcp_server *server,
				       const struct xiaozhi_mcp_tool *tools,
				       size_t tool_count);

int xiaozhi_mcp_register_default_tools(struct xiaozhi_mcp_server *server);

cJSON *xiaozhi_mcp_result_text(const char *text);
cJSON *xiaozhi_mcp_result_bool(int value);
cJSON *xiaozhi_mcp_result_json(const cJSON *value);

int xiaozhi_mcp_handle(struct xiaozhi_mcp_server *server,
			       const cJSON *payload,
			       xiaozhi_mcp_send_fn send,
			       void *send_opaque);

#endif
