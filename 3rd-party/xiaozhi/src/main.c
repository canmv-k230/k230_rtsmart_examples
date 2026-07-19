#include "xiaozhi_app.h"
#include "xiaozhi_mpp.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t interrupted;

static void signal_handler(int sig)
{
	(void)sig;
	interrupted = 1;
}

static int copy_string(char *dst, size_t dst_size, const char *src,
			       const char *name)
{
	int written = snprintf(dst, dst_size, "%s", src ? src : "");

	if (written < 0 || (size_t)written >= dst_size) {
		printf("%s is too long\n", name);
		return -1;
	}
	return 0;
}

static int parse_int(const char *arg, int min, int max, int *value)
{
	char *end = NULL;
	long parsed;

	parsed = strtol(arg, &end, 10);
	if (!arg[0] || (end && *end) || parsed < min || parsed > max)
		return -1;
	*value = (int)parsed;
	return 0;
}

static int parse_float(const char *arg, float min, float max, float *value)
{
	char *end = NULL;
	float parsed;

	parsed = strtof(arg, &end);
	if (!arg[0] || (end && *end) || parsed != parsed || parsed < min ||
	    parsed > max)
		return -1;
	*value = parsed;
	return 0;
}

static int parse_log_level(const char *arg, int *value)
{
	if (!strcmp(arg, "error"))
		*value = XIAOZHI_LOG_LEVEL_ERROR;
	else if (!strcmp(arg, "warn") || !strcmp(arg, "warning"))
		*value = XIAOZHI_LOG_LEVEL_WARN;
	else if (!strcmp(arg, "info"))
		*value = XIAOZHI_LOG_LEVEL_INFO;
	else if (!strcmp(arg, "debug"))
		*value = XIAOZHI_LOG_LEVEL_DEBUG;
	else
		return -1;
	return 0;
}

static int parse_url(struct xiaozhi_app_config *config, const char *url)
{
	const char *authority;
	const char *path;
	const char *port_start;
	size_t authority_len;
	size_t host_len;
	size_t port_len;
	char port_buffer[8];
	int default_port;
	int written;

	if (!strncmp(url, "wss://", 6)) {
		authority = url + 6;
		default_port = 443;
		config->use_ssl = 1;
	} else if (!strncmp(url, "ws://", 5)) {
		authority = url + 5;
		default_port = 80;
		config->use_ssl = 0;
	} else {
		printf("WebSocket URL must start with ws:// or wss://\n");
		return -1;
	}

	path = strchr(authority, '/');
	authority_len = path ? (size_t)(path - authority) : strlen(authority);
	port_start = memchr(authority, ':', authority_len);
	host_len = port_start ? (size_t)(port_start - authority) : authority_len;
	if (!host_len || host_len >= sizeof(config->address)) {
		printf("WebSocket host is invalid or too long\n");
		return -1;
	}
	memcpy(config->address, authority, host_len);
	config->address[host_len] = '\0';

	config->port = default_port;
	if (port_start) {
		port_len = (size_t)(authority + authority_len - port_start - 1);
		if (!port_len || port_len >= sizeof(port_buffer)) {
			printf("WebSocket URL port is invalid\n");
			return -1;
		}
		memcpy(port_buffer, port_start + 1, port_len);
		port_buffer[port_len] = '\0';
		if (parse_int(port_buffer, 1, 65535, &config->port)) {
			printf("WebSocket URL port is invalid\n");
			return -1;
		}
	}

	written = snprintf(config->path, sizeof(config->path), "%s",
			   path ? path : "/");
	if (written < 0 || (size_t)written >= sizeof(config->path)) {
		printf("WebSocket path is too long\n");
		return -1;
	}
	config->ssl_explicit = 1;
	return 0;
}

static void print_usage(const char *program)
{
	printf("Usage: %s [options]\n", program);
	printf("\nConnect to xiaozhi, answer MCP requests, and stream K230 Opus audio.\n");
	printf("Options:\n");
	printf("      --url <url>             WebSocket URL\n");
	printf("  -a, --address <host>       Server host (default: %s)\n",
	       XIAOZHI_DEFAULT_ADDRESS);
	printf("  -p, --port <port>          Server port (default: %d)\n",
	       XIAOZHI_DEFAULT_PORT);
	printf("  -u, --path <path>          WebSocket path (default: %s)\n",
	       XIAOZHI_DEFAULT_PATH);
	printf("  -s, --ssl                  Use wss://\n");
	printf("  -n, --no-ssl               Use ws://\n");
	printf("  -i, --insecure             Skip TLS certificate verification (default)\n");
	printf("  -v, --verify-tls           Verify TLS certificates\n");
	printf("  -k, --token <token>        Token or complete Authorization value\n");
	printf("      --activation-url <url> OTA activation endpoint\n");
	printf("      --no-activation        Skip OTA activation\n");
	printf("      --device-id <id>       Device-Id header value\n");
	printf("      --client-id <id>       Client-Id header value\n");
	printf("  -t, --timeout <seconds>    Connect and hello timeout (default: %d)\n",
	       XIAOZHI_DEFAULT_TIMEOUT_SECS);
	printf("  -r, --attempts <count>     Maximum attempts; 0 retries forever (default)\n");
	printf("  -d, --duration <seconds>   Stop after a ready session; 0 keeps it open\n");
	printf("      --log-level <level>    error, warn, info, or debug (default: info)\n");
	printf("      --debug                Same as --log-level debug\n");
	printf("      --no-audio             Transport and MCP test without K230 audio\n");
	printf("      --no-lvgl              Disable the LVGL status UI\n");
	printf("      --lvgl-connector <id>  K230 display connector type (default: ST7701 480x800)\n");
	printf("      --lvgl-layer <id>      K230 OSD layer (default: OSD0)\n");
	printf("      --lvgl-touch <id>      Touch device id (default: 0)\n");
	printf("      --lvgl-resource-dir <dir> Font and emotion image/GIF directory\n");
	printf("      --input-device <name>  i2s (default) or pdm\n");
	printf("      --input-channel <n>    AI channel (default: 0)\n");
	printf("      --output-channel <n>   AO channel (default: 0)\n");
	printf("      --external-codec       Use an external I2S codec\n");
	printf("      --audio3a <mask>       1=ANS, 2=AGC, 4=AEC (combine bits)\n");
	printf("      --ans                  Enable microphone noise suppression\n");
	printf("      --agc                  Enable microphone automatic gain control\n");
	printf("      --aec                  Enable microphone acoustic echo cancellation\n");
	printf("      --mode <mode>          realtime (default), auto, or manual\n");
	printf("      --wake-word <model>    Wait for a local wake word before realtime\n");
	printf("      --no-wake-word         Start the configured listen mode immediately\n");
	printf("      --wake-word-task <id>  KWS task (default: %s)\n",
	       XIAOZHI_DEFAULT_WAKE_WORD_TASK);
	printf("      --wake-word-text <txt> Protocol wake-word text (default: %s)\n",
	       XIAOZHI_DEFAULT_WAKE_WORD_TEXT);
	printf("      --wake-word-keywords <n> KWS output count (default: %d)\n",
	       XIAOZHI_DEFAULT_WAKE_WORD_KEYWORDS);
	printf("      --wake-word-threshold <f> KWS threshold (default: %.2f)\n",
	       XIAOZHI_DEFAULT_WAKE_WORD_THRESHOLD);
	printf("  -h, --help                 Show this help\n");
}

static int parse_args(struct xiaozhi_app_config *config, int argc, char **argv)
{
	int i;

	for (i = 1; i < argc; i++) {
		const char *arg = argv[i];
		const char *value;

		if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
			print_usage(argv[0]);
			return 1;
		} else if (!strcmp(arg, "--url")) {
			if (++i >= argc || parse_url(config, argv[i]))
				return -1;
		} else if (!strcmp(arg, "-a") || !strcmp(arg, "--address")) {
			if (++i >= argc || copy_string(config->address,
						      sizeof(config->address), argv[i], "address"))
				return -1;
		} else if (!strcmp(arg, "-p") || !strcmp(arg, "--port")) {
			int ssl_explicit = config->ssl_explicit;

			if (++i >= argc || parse_int(argv[i], 1, 65535, &config->port))
				return -1;
			if (!ssl_explicit)
				config->use_ssl = config->port == 443;
		} else if (!strcmp(arg, "-u") || !strcmp(arg, "--path")) {
			if (++i >= argc || argv[i][0] != '/' ||
			    copy_string(config->path, sizeof(config->path), argv[i], "path"))
				return -1;
		} else if (!strcmp(arg, "-s") || !strcmp(arg, "--ssl")) {
			config->use_ssl = 1;
			config->ssl_explicit = 1;
			if (config->port == 80)
				config->port = 443;
		} else if (!strcmp(arg, "-n") || !strcmp(arg, "--no-ssl")) {
			config->use_ssl = 0;
			config->ssl_explicit = 1;
			if (config->port == 443)
				config->port = 80;
		} else if (!strcmp(arg, "-i") || !strcmp(arg, "--insecure")) {
			config->allow_insecure = 1;
		} else if (!strcmp(arg, "-v") || !strcmp(arg, "--verify-tls")) {
			config->allow_insecure = 0;
		} else if (!strcmp(arg, "-k") || !strcmp(arg, "--token")) {
			if (++i >= argc || copy_string(config->token, sizeof(config->token),
						      argv[i], "token"))
				return -1;
			config->token_explicit = 1;
		} else if (!strcmp(arg, "--activation-url")) {
			if (++i >= argc || copy_string(config->activation_url,
						      sizeof(config->activation_url), argv[i],
						      "activation url"))
				return -1;
			config->activation_enabled = 1;
		} else if (!strcmp(arg, "--no-activation")) {
			config->activation_enabled = 0;
		} else if (!strcmp(arg, "--device-id")) {
			if (++i >= argc || copy_string(config->device_id,
						      sizeof(config->device_id), argv[i], "device id"))
				return -1;
			config->device_id_explicit = 1;
		} else if (!strcmp(arg, "--client-id")) {
			if (++i >= argc || copy_string(config->client_id,
						      sizeof(config->client_id), argv[i], "client id"))
				return -1;
			config->client_id_explicit = 1;
		} else if (!strcmp(arg, "-t") || !strcmp(arg, "--timeout")) {
			if (++i >= argc || parse_int(argv[i], 1, 3600,
							&config->timeout_secs))
				return -1;
		} else if (!strcmp(arg, "-r") || !strcmp(arg, "--attempts")) {
			if (++i >= argc || parse_int(argv[i], 0, 1000,
							&config->max_attempts))
				return -1;
		} else if (!strcmp(arg, "-d") || !strcmp(arg, "--duration")) {
			if (++i >= argc || parse_int(argv[i], 0, 86400,
						&config->duration_secs))
				return -1;
		} else if (!strcmp(arg, "--log-level")) {
			if (++i >= argc || parse_log_level(argv[i], &config->log_level))
				return -1;
		} else if (!strcmp(arg, "--debug")) {
			config->log_level = XIAOZHI_LOG_LEVEL_DEBUG;
		} else if (!strcmp(arg, "--no-audio")) {
			config->audio_enabled = 0;
			config->wake_word_enabled = 0;
		} else if (!strcmp(arg, "--no-lvgl")) {
			config->lvgl_enabled = 0;
		} else if (!strcmp(arg, "--lvgl-connector")) {
			if (++i >= argc || parse_int(argv[i], 1, 0x7fffffff,
						&config->lvgl_connector))
				return -1;
		} else if (!strcmp(arg, "--lvgl-layer")) {
			if (++i >= argc || parse_int(argv[i], 4, 7,
						&config->lvgl_layer))
				return -1;
		} else if (!strcmp(arg, "--lvgl-touch")) {
			if (++i >= argc || parse_int(argv[i], 0, 3,
						&config->lvgl_touch_id))
				return -1;
		} else if (!strcmp(arg, "--lvgl-resource-dir")) {
			if (++i >= argc || copy_string(config->lvgl_resource_dir,
						      sizeof(config->lvgl_resource_dir), argv[i],
						      "LVGL resource directory"))
				return -1;
		} else if (!strcmp(arg, "--input-device")) {
			if (++i >= argc)
				return -1;
			value = argv[i];
			if (!strcmp(value, "i2s"))
				config->audio_input_device = 0;
			else if (!strcmp(value, "pdm"))
				config->audio_input_device = 1;
			else
				return -1;
		} else if (!strcmp(arg, "--input-channel")) {
			if (++i >= argc || parse_int(argv[i], 0, 3,
							&config->audio_input_channel))
				return -1;
		} else if (!strcmp(arg, "--output-channel")) {
			if (++i >= argc || parse_int(argv[i], 0, 1,
							&config->audio_output_channel))
				return -1;
		} else if (!strcmp(arg, "--external-codec")) {
			config->audio_internal_codec = 0;
		} else if (!strcmp(arg, "--audio3a")) {
			config->audio3a_explicit = 1;
			if (++i >= argc || parse_int(argv[i], 0,
						XIAOZHI_AUDIO_3A_ALL,
						&config->audio3a_mask))
				return -1;
		} else if (!strcmp(arg, "--ans")) {
			config->audio3a_explicit = 1;
			config->audio3a_mask |= XIAOZHI_AUDIO_3A_ANS;
		} else if (!strcmp(arg, "--agc")) {
			config->audio3a_explicit = 1;
			config->audio3a_mask |= XIAOZHI_AUDIO_3A_AGC;
		} else if (!strcmp(arg, "--aec")) {
			config->audio3a_explicit = 1;
			config->audio3a_mask |= XIAOZHI_AUDIO_3A_AEC;
		} else if (!strcmp(arg, "--mode")) {
			if (++i >= argc)
				return -1;
			value = argv[i];
			if (!strcmp(value, "auto"))
				config->mode = XIAOZHI_MODE_AUTO;
			else if (!strcmp(value, "manual"))
				config->mode = XIAOZHI_MODE_MANUAL;
			else if (!strcmp(value, "realtime"))
				config->mode = XIAOZHI_MODE_REALTIME;
			else
				return -1;
		} else if (!strcmp(arg, "--wake-word")) {
			if (++i >= argc || copy_string(config->wake_word_model,
						      sizeof(config->wake_word_model), argv[i],
						      "wake-word model"))
				return -1;
			config->wake_word_enabled = 1;
		} else if (!strcmp(arg, "--no-wake-word")) {
			config->wake_word_enabled = 0;
		} else if (!strcmp(arg, "--wake-word-task")) {
			if (++i >= argc || copy_string(config->wake_word_task,
						      sizeof(config->wake_word_task), argv[i],
						      "wake-word task"))
				return -1;
		} else if (!strcmp(arg, "--wake-word-text")) {
			if (++i >= argc || copy_string(config->wake_word_text,
						      sizeof(config->wake_word_text), argv[i],
						      "wake-word text"))
				return -1;
		} else if (!strcmp(arg, "--wake-word-keywords")) {
			if (++i >= argc || parse_int(argv[i], 1, 32,
						&config->wake_word_keywords))
				return -1;
		} else if (!strcmp(arg, "--wake-word-threshold")) {
			if (++i >= argc || parse_float(argv[i], 0.0f, 1.0f,
						  &config->wake_word_threshold))
				return -1;
		} else {
			printf("unknown option: %s\n", arg);
			return -1;
		}
	}
	if (config->mode == XIAOZHI_MODE_REALTIME && !config->audio3a_explicit)
		config->audio3a_mask = XIAOZHI_AUDIO_3A_DEFAULT;
	if (!config->audio_enabled)
		config->wake_word_enabled = 0;
	if (config->wake_word_enabled && config->mode != XIAOZHI_MODE_REALTIME) {
		printf("--wake-word requires --mode realtime\n");
		return -1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	struct xiaozhi_app_config config;
	int ret;

	xiaozhi_app_config_init(&config);
	ret = parse_args(&config, argc, argv);
	if (ret > 0)
		return 0;
	if (ret < 0) {
		print_usage(argv[0]);
		return 1;
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	if (xiaozhi_mpp_initialize())
		return 1;
	ret = xiaozhi_app_run(&config, &interrupted);
	xiaozhi_mpp_deinitialize();
	return ret;
}
