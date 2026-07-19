#include "xiaozhi_light.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT

#include "drv_fpioa.h"
#include "drv_gpio.h"

struct xiaozhi_light {
	drv_gpio_inst_t *gpio;
	int pin;
	int on_value;
	int on;
};

static gpio_pin_value_t light_gpio_value(const struct xiaozhi_light *light,
						 int on)
{
	if (on)
		return light->on_value ? GPIO_PV_HIGH : GPIO_PV_LOW;
	return light->on_value ? GPIO_PV_LOW : GPIO_PV_HIGH;
}

#endif

void xiaozhi_light_config_init(struct xiaozhi_light_config *config)
{
	if (!config)
		return;
	memset(config, 0, sizeof(*config));
	config->pin = -1;

#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT
	config->enabled = 1;
	config->pin = CONFIG_RTSMART_XIAOZHI_MCP_LIGHT_GPIO;
	config->on_value = CONFIG_RTSMART_XIAOZHI_MCP_LIGHT_ON_VALUE;
#endif
}

struct xiaozhi_light *xiaozhi_light_create(
	const struct xiaozhi_light_config *config)
{
#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT
	struct xiaozhi_light *light;

	if (!config || !config->enabled || config->pin < 0 ||
	    config->pin >= GPIO_MAX_NUM ||
	    (config->on_value != 0 && config->on_value != 1))
		return NULL;

	light = calloc(1, sizeof(*light));
	if (!light)
		return NULL;
	light->pin = config->pin;
	light->on_value = config->on_value;

	if (drv_fpioa_set_pin_func(light->pin,
					   (fpioa_func_t)(GPIO0 + light->pin)) != 0 ||
	    drv_gpio_inst_create(light->pin, &light->gpio) != 0 ||
	    drv_gpio_mode_set(light->gpio, GPIO_DM_OUTPUT) != 0 ||
	    drv_gpio_value_set(light->gpio, light_gpio_value(light, 0)) != 0) {
		printf("xiaozhi: failed to initialize GPIO light on GPIO%d\n",
		       light->pin);
		if (light->gpio)
			drv_gpio_inst_destroy(&light->gpio);
		free(light);
		return NULL;
	}

	printf("xiaozhi: GPIO light enabled on GPIO%d (on value %d)\n",
	       light->pin, light->on_value);
	return light;
#else
	(void)config;
	return NULL;
#endif
}

void xiaozhi_light_destroy(struct xiaozhi_light **light)
{
#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT
	if (!light || !*light)
		return;
	if ((*light)->gpio) {
		drv_gpio_value_set((*light)->gpio,
					light_gpio_value(*light, 0));
		drv_gpio_inst_destroy(&(*light)->gpio);
	}
	free(*light);
	*light = NULL;
#else
	(void)light;
#endif
}

int xiaozhi_light_set(struct xiaozhi_light *light, int on)
{
#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT
	if (!light || !light->gpio)
		return -1;
	if (drv_gpio_value_set(light->gpio,
				       light_gpio_value(light, on != 0)) != 0)
		return -1;
	light->on = on != 0;
	return 0;
#else
	(void)light;
	(void)on;
	return -1;
#endif
}

int xiaozhi_light_get(struct xiaozhi_light *light, int *on)
{
#if defined(CONFIG_RTSMART_XIAOZHI_MCP_LIGHT) && \
	CONFIG_RTSMART_XIAOZHI_MCP_LIGHT
	if (!light || !light->gpio || !on)
		return -1;
	*on = light->on;
	return 0;
#else
	(void)light;
	(void)on;
	return -1;
#endif
}
