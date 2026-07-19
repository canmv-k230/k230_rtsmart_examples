#include "xiaozhi_mcp_internal.h"

#include <stdio.h>
#include <string.h>

#define XIAOZHI_MCP_MAX_LIST_BYTES 8192
#define XIAOZHI_MCP_MAX_ERROR 160

static int send_json(cJSON *root, xiaozhi_mcp_send_fn send, void *opaque)
{
	char *json;
	int ret;

	if (!root || !send)
		return -1;
	json = cJSON_PrintUnformatted(root);
	if (!json)
		return -1;
	ret = send(opaque, json, strlen(json));
	cJSON_free(json);
	return ret;
}

static int reply(const cJSON *id, cJSON *result, xiaozhi_mcp_send_fn send,
		 void *opaque)
{
	cJSON *root = NULL;
	cJSON *id_copy = NULL;
	int ret;

	if (!id || !result || !send)
		goto failure;
	root = cJSON_CreateObject();
	id_copy = cJSON_Duplicate(id, 1);
	if (!root || !id_copy)
		goto failure;
	cJSON_AddStringToObject(root, "jsonrpc", "2.0");
	cJSON_AddItemToObject(root, "id", id_copy);
	id_copy = NULL;
	cJSON_AddItemToObject(root, "result", result);
	result = NULL;
	ret = send_json(root, send, opaque);
	cJSON_Delete(root);
	return ret;

failure:
	if (id_copy)
		cJSON_Delete(id_copy);
	if (root)
		cJSON_Delete(root);
	if (result)
		cJSON_Delete(result);
	return -1;
}

static int reply_error(const cJSON *id, int code, const char *message,
			       xiaozhi_mcp_send_fn send, void *opaque)
{
	cJSON *root = NULL;
	cJSON *error = NULL;
	cJSON *id_copy = NULL;
	int ret;

	if (!send)
		return -1;
	root = cJSON_CreateObject();
	error = cJSON_CreateObject();
	if (!root || !error)
		goto failure;
	if (id) {
		id_copy = cJSON_Duplicate(id, 1);
		if (!id_copy)
			goto failure;
		cJSON_AddItemToObject(root, "id", id_copy);
		id_copy = NULL;
	} else {
		cJSON_AddNullToObject(root, "id");
	}
	cJSON_AddStringToObject(root, "jsonrpc", "2.0");
	cJSON_AddNumberToObject(error, "code", code);
	cJSON_AddStringToObject(error, "message", message ? message : "error");
	cJSON_AddItemToObject(root, "error", error);
	error = NULL;
	ret = send_json(root, send, opaque);
	cJSON_Delete(root);
	return ret;

failure:
	if (id_copy)
		cJSON_Delete(id_copy);
	if (root)
		cJSON_Delete(root);
	if (error)
		cJSON_Delete(error);
	return -1;
}

void xiaozhi_mcp_set_error(char *error, size_t error_size, const char *message)
{
	if (!error || !error_size)
		return;
	snprintf(error, error_size, "%s", message ? message : "tool failed");
}

static cJSON *make_text_content(const char *text)
{
	cJSON *item = cJSON_CreateObject();

	if (!item)
		return NULL;
	cJSON_AddStringToObject(item, "type", "text");
	cJSON_AddStringToObject(item, "text", text ? text : "");
	return item;
}

cJSON *xiaozhi_mcp_result_text(const char *text)
{
	cJSON *result = NULL;
	cJSON *content = NULL;
	cJSON *item = NULL;

	result = cJSON_CreateObject();
	content = cJSON_CreateArray();
	item = make_text_content(text);
	if (!result || !content || !item)
		goto failure;
	cJSON_AddItemToArray(content, item);
	item = NULL;
	cJSON_AddItemToObject(result, "content", content);
	content = NULL;
	cJSON_AddBoolToObject(result, "isError", 0);
	return result;

failure:
	if (item)
		cJSON_Delete(item);
	if (content)
		cJSON_Delete(content);
	if (result)
		cJSON_Delete(result);
	return NULL;
}

cJSON *xiaozhi_mcp_result_bool(int value)
{
	return xiaozhi_mcp_result_text(value ? "true" : "false");
}

cJSON *xiaozhi_mcp_result_json(const cJSON *value)
{
	char *json;
	cJSON *result;

	if (!value)
		return NULL;
	json = cJSON_PrintUnformatted(value);
	if (!json)
		return NULL;
	result = xiaozhi_mcp_result_text(json);
	cJSON_free(json);
	return result;
}

static const char *property_type_name(enum xiaozhi_mcp_property_type type)
{
	switch (type) {
	case XIAOZHI_MCP_PROPERTY_BOOLEAN:
		return "boolean";
	case XIAOZHI_MCP_PROPERTY_INTEGER:
		return "integer";
	case XIAOZHI_MCP_PROPERTY_NUMBER:
		return "number";
	case XIAOZHI_MCP_PROPERTY_STRING:
		return "string";
	case XIAOZHI_MCP_PROPERTY_OBJECT:
		return "object";
	case XIAOZHI_MCP_PROPERTY_ARRAY:
		return "array";
	default:
		return NULL;
	}
}

static cJSON *make_input_schema(const struct xiaozhi_mcp_tool *tool)
{
	cJSON *schema = NULL;
	cJSON *properties = NULL;
	cJSON *required = NULL;
	size_t i;

	schema = cJSON_CreateObject();
	properties = cJSON_CreateObject();
	if (!schema || !properties)
		goto failure;
	cJSON_AddStringToObject(schema, "type", "object");
	for (i = 0; i < tool->property_count; i++) {
		const struct xiaozhi_mcp_property *property =
			&tool->properties[i];
		cJSON *definition = cJSON_CreateObject();
		const char *type = property_type_name(property->type);

		if (!definition || !type || !property->name) {
			if (definition)
				cJSON_Delete(definition);
			goto failure;
		}
		cJSON_AddStringToObject(definition, "type", type);
		if (property->description)
			cJSON_AddStringToObject(definition, "description",
						property->description);
		if (property->has_minimum)
			cJSON_AddNumberToObject(definition, "minimum",
						property->minimum);
		if (property->has_maximum)
			cJSON_AddNumberToObject(definition, "maximum",
						property->maximum);
		cJSON_AddItemToObject(properties, property->name, definition);
		if (property->required) {
			if (!required) {
				required = cJSON_CreateArray();
				if (!required)
					goto failure;
			}
			cJSON_AddItemToArray(required,
					cJSON_CreateString(property->name));
		}
	}
	cJSON_AddItemToObject(schema, "properties", properties);
	properties = NULL;
	if (required) {
		cJSON_AddItemToObject(schema, "required", required);
		required = NULL;
	}
	return schema;

failure:
	if (required)
		cJSON_Delete(required);
	if (properties)
		cJSON_Delete(properties);
	if (schema)
		cJSON_Delete(schema);
	return NULL;
}

static cJSON *make_tool_description(const struct xiaozhi_mcp_tool *tool)
{
	cJSON *description;
	cJSON *schema;
	cJSON *annotations;
	cJSON *audience;

	if (!tool || !tool->name || !tool->description)
		return NULL;
	description = cJSON_CreateObject();
	schema = make_input_schema(tool);
	if (!description || !schema) {
		if (description)
			cJSON_Delete(description);
		if (schema)
			cJSON_Delete(schema);
		return NULL;
	}
	cJSON_AddStringToObject(description, "name", tool->name);
	cJSON_AddStringToObject(description, "description", tool->description);
	cJSON_AddItemToObject(description, "inputSchema", schema);
	if (tool->user_only) {
		annotations = cJSON_CreateObject();
		audience = cJSON_CreateArray();
		if (!annotations || !audience) {
			if (annotations)
				cJSON_Delete(annotations);
			if (audience)
				cJSON_Delete(audience);
			cJSON_Delete(description);
			return NULL;
		}
		cJSON_AddItemToArray(audience, cJSON_CreateString("user"));
		cJSON_AddItemToObject(annotations, "audience", audience);
		cJSON_AddItemToObject(description, "annotations", annotations);
	}
	return description;
}

static cJSON *make_initialize_result(const struct xiaozhi_mcp_device *device)
{
	cJSON *result = NULL;
	cJSON *capabilities = NULL;
	cJSON *tools = NULL;
	cJSON *server_info = NULL;

	result = cJSON_CreateObject();
	capabilities = cJSON_CreateObject();
	tools = cJSON_CreateObject();
	server_info = cJSON_CreateObject();
	if (!result || !capabilities || !tools || !server_info)
		goto failure;
	cJSON_AddStringToObject(result, "protocolVersion", "2024-11-05");
	cJSON_AddItemToObject(capabilities, "tools", tools);
	tools = NULL;
	cJSON_AddItemToObject(result, "capabilities", capabilities);
	capabilities = NULL;
	cJSON_AddStringToObject(server_info, "name",
				device && device->name ? device->name : "canmv-k230");
	cJSON_AddStringToObject(server_info, "version",
				device && device->version ? device->version : "1.0.0");
	cJSON_AddItemToObject(result, "serverInfo", server_info);
	server_info = NULL;
	return result;

failure:
	if (server_info)
		cJSON_Delete(server_info);
	if (tools)
		cJSON_Delete(tools);
	if (capabilities)
		cJSON_Delete(capabilities);
	if (result)
		cJSON_Delete(result);
	return NULL;
}

static int is_valid_id(const cJSON *id)
{
	return cJSON_IsNumber(id) || cJSON_IsString(id);
}

static int validate_number(const cJSON *value,
			   const struct xiaozhi_mcp_property *property)
{
	double number;

	if (!cJSON_IsNumber(value))
		return -1;
	number = value->valuedouble;
	if (number != number)
		return -1;
	if (property->type == XIAOZHI_MCP_PROPERTY_INTEGER) {
		if (number < -2147483648.0 || number > 2147483647.0 ||
		    (double)(int)number != number)
			return -1;
	}
	if (property->has_minimum && number < property->minimum)
		return -1;
	if (property->has_maximum && number > property->maximum)
		return -1;
	return 0;
}

static int validate_arguments(const struct xiaozhi_mcp_tool *tool,
			      const cJSON *arguments, char *error, size_t error_size)
{
	size_t i;

	if (arguments && !cJSON_IsObject(arguments)) {
		xiaozhi_mcp_set_error(error, error_size, "Invalid arguments");
		return -1;
	}
	for (i = 0; i < tool->property_count; i++) {
		const struct xiaozhi_mcp_property *property = &tool->properties[i];
		const cJSON *value = arguments ?
			cJSON_GetObjectItemCaseSensitive(arguments, property->name) : NULL;
		int valid = 0;

		if (!value) {
			if (!property->required)
				continue;
			if (error && error_size)
				snprintf(error, error_size, "Missing argument: %s",
					 property->name);
			return -1;
		}
		switch (property->type) {
		case XIAOZHI_MCP_PROPERTY_BOOLEAN:
			valid = cJSON_IsBool(value);
			break;
		case XIAOZHI_MCP_PROPERTY_INTEGER:
		case XIAOZHI_MCP_PROPERTY_NUMBER:
			valid = validate_number(value, property) == 0;
			break;
		case XIAOZHI_MCP_PROPERTY_STRING:
			valid = cJSON_IsString(value);
			break;
		case XIAOZHI_MCP_PROPERTY_OBJECT:
			valid = cJSON_IsObject(value);
			break;
		case XIAOZHI_MCP_PROPERTY_ARRAY:
			valid = cJSON_IsArray(value);
			break;
		default:
			break;
		}
		if (!valid) {
			if (error && error_size)
				snprintf(error, error_size, "Invalid argument: %s",
					 property->name);
			return -1;
		}
	}
	return 0;
}

static int make_tools_result(const struct xiaozhi_mcp_server *server,
			     const char *cursor, int with_user_tools, cJSON **result,
			     char *error, size_t error_size)
{
	cJSON *response = NULL;
	cJSON *tools = NULL;
	const char *next_cursor = NULL;
	int cursor_found;
	size_t i;

	if (!server || !result)
		return -1;
	response = cJSON_CreateObject();
	tools = cJSON_CreateArray();
	if (!response || !tools)
		goto failure;
	cursor_found = !cursor || !cursor[0];
	for (i = 0; i < server->tool_count; i++) {
		const struct xiaozhi_mcp_tool *tool = server->tools[i];
		cJSON *description;
		char *encoded;

		if (!cursor_found) {
			if (!strcmp(tool->name, cursor))
				cursor_found = 1;
			else
				continue;
		}
		if (tool->user_only && !with_user_tools)
			continue;
		description = make_tool_description(tool);
		if (!description)
			goto failure;
		cJSON_AddItemToArray(tools, description);
		encoded = cJSON_PrintUnformatted(response);
		if (!encoded)
			goto failure;
		if (strlen(encoded) > XIAOZHI_MCP_MAX_LIST_BYTES) {
			cJSON_DeleteItemFromArray(tools,
						 cJSON_GetArraySize(tools) - 1);
			next_cursor = tool->name;
			cJSON_free(encoded);
			break;
		}
		cJSON_free(encoded);
	}
	if (!cursor_found) {
		xiaozhi_mcp_set_error(error, error_size, "Invalid tools/list cursor");
		goto failure;
	}
	cJSON_AddItemToObject(response, "tools", tools);
	tools = NULL;
	if (next_cursor)
		cJSON_AddStringToObject(response, "nextCursor", next_cursor);
	*result = response;
	return 0;

failure:
	if (tools)
		cJSON_Delete(tools);
	if (response)
		cJSON_Delete(response);
	return -1;
}

static const struct xiaozhi_mcp_tool *find_tool(
		const struct xiaozhi_mcp_server *server, const char *name)
{
	size_t i;

	for (i = 0; i < server->tool_count; i++)
		if (!strcmp(server->tools[i]->name, name))
			return server->tools[i];
	return NULL;
}

int xiaozhi_mcp_server_init(struct xiaozhi_mcp_server *server,
				    const struct xiaozhi_mcp_device *device)
{
	if (!server)
		return -1;
	memset(server, 0, sizeof(*server));
	server->device = device;
	return 0;
}

int xiaozhi_mcp_register_tool(struct xiaozhi_mcp_server *server,
				      const struct xiaozhi_mcp_tool *tool)
{
	size_t i;
	size_t j;

	if (!server || !tool || !tool->name || !tool->description || !tool->call ||
	    (tool->property_count > 0 && !tool->properties))
		return -1;
	for (i = 0; i < tool->property_count; i++) {
		const struct xiaozhi_mcp_property *property = &tool->properties[i];

		if (!property->name || !property_type_name(property->type) ||
		    (property->has_minimum && property->has_maximum &&
		     property->minimum > property->maximum))
			return -1;
		for (j = 0; j < i; j++)
			if (!strcmp(tool->properties[j].name, property->name))
				return -1;
	}
	for (i = 0; i < server->tool_count; i++)
		if (!strcmp(server->tools[i]->name, tool->name))
			return -2;
	if (server->tool_count >= XIAOZHI_MCP_MAX_TOOLS)
		return -1;
	server->tools[server->tool_count++] = tool;
	return 0;
}

int xiaozhi_mcp_register_tools(struct xiaozhi_mcp_server *server,
				       const struct xiaozhi_mcp_tool *tools,
				       size_t tool_count)
{
	size_t i;

	if (!server || (!tools && tool_count))
		return -1;
	for (i = 0; i < tool_count; i++)
		if (xiaozhi_mcp_register_tool(server, &tools[i]))
			return -1;
	return 0;
}

int xiaozhi_mcp_handle(struct xiaozhi_mcp_server *server,
			       const cJSON *payload,
			       xiaozhi_mcp_send_fn send,
			       void *send_opaque)
{
	const cJSON *version;
	const cJSON *method;
	const cJSON *id;
	const cJSON *params;
	const cJSON *value;
	cJSON *result = NULL;
	char error[XIAOZHI_MCP_MAX_ERROR];

	if (!server || !cJSON_IsObject(payload) || !send)
		return -1;
	version = cJSON_GetObjectItemCaseSensitive(payload, "jsonrpc");
	method = cJSON_GetObjectItemCaseSensitive(payload, "method");
	if (!cJSON_IsString(version) || strcmp(version->valuestring, "2.0") ||
	    !cJSON_IsString(method))
		return reply_error(NULL, -32600, "Invalid JSON-RPC request", send,
				   send_opaque);
	if (!strncmp(method->valuestring, "notifications/", 13))
		return 0;
	id = cJSON_GetObjectItemCaseSensitive(payload, "id");
	if (!is_valid_id(id))
		return reply_error(NULL, -32600, "Request id is required", send,
				   send_opaque);
	params = cJSON_GetObjectItemCaseSensitive(payload, "params");
	if (params && !cJSON_IsObject(params))
		return reply_error(id, -32602, "Invalid params", send, send_opaque);

	if (!strcmp(method->valuestring, "initialize")) {
		result = make_initialize_result(server->device);
		if (!result)
			return reply_error(id, -32000, "Failed to initialize MCP", send,
					   send_opaque);
		return reply(id, result, send, send_opaque);
	}

	if (!strcmp(method->valuestring, "tools/list")) {
		const cJSON *cursor = params ?
			cJSON_GetObjectItemCaseSensitive(params, "cursor") : NULL;
		const cJSON *with_user_tools = params ?
			cJSON_GetObjectItemCaseSensitive(params, "withUserTools") : NULL;
		int with_user = 0;

		if (cursor && !cJSON_IsString(cursor))
			return reply_error(id, -32602, "cursor must be a string", send,
					   send_opaque);
		if (with_user_tools) {
			if (!cJSON_IsBool(with_user_tools))
				return reply_error(id, -32602,
						   "withUserTools must be a boolean", send,
						   send_opaque);
			with_user = cJSON_IsTrue(with_user_tools);
		}
		error[0] = '\0';
		if (make_tools_result(server, cursor ? cursor->valuestring : NULL,
					      with_user, &result, error, sizeof(error)))
			return reply_error(id, -32602, error[0] ? error : "Failed to list tools",
					   send, send_opaque);
		return reply(id, result, send, send_opaque);
	}

	if (!strcmp(method->valuestring, "tools/call")) {
		const cJSON *name;
		const cJSON *arguments;
		const struct xiaozhi_mcp_tool *tool;

		if (!params)
			return reply_error(id, -32602, "Missing params", send,
					   send_opaque);
		name = cJSON_GetObjectItemCaseSensitive(params, "name");
		arguments = cJSON_GetObjectItemCaseSensitive(params, "arguments");
		if (!cJSON_IsString(name))
			return reply_error(id, -32602, "Missing tool name", send,
					   send_opaque);
		if (arguments && !cJSON_IsObject(arguments))
			return reply_error(id, -32602, "Invalid arguments", send,
					   send_opaque);
		tool = find_tool(server, name->valuestring);
		if (!tool)
			return reply_error(id, -32601, "Unknown tool", send, send_opaque);
		printf("xiaozhi: MCP tool call: %s\n", name->valuestring);
		error[0] = '\0';
		if (validate_arguments(tool, arguments, error, sizeof(error)))
			return reply_error(id, -32602,
					   error[0] ? error : "Invalid arguments", send,
					   send_opaque);
		error[0] = '\0';
		value = arguments;
		if (tool->call(tool->opaque ? tool->opaque : (void *)server->device,
				       value, &result, error, sizeof(error)))
			return reply_error(id, -32000,
					   error[0] ? error : "Tool call failed", send,
					   send_opaque);
		if (!result)
			return reply_error(id, -32000, "Tool returned no result", send,
					   send_opaque);
		return reply(id, result, send, send_opaque);
	}

	return reply_error(id, -32601, "Method not implemented", send,
			   send_opaque);
}
