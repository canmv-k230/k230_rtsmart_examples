#ifndef XIAOZHI_LIGHT_H
#define XIAOZHI_LIGHT_H

struct xiaozhi_light_config {
	int enabled;
	int pin;
	int on_value;
};

struct xiaozhi_light;

void xiaozhi_light_config_init(struct xiaozhi_light_config *config);

struct xiaozhi_light *xiaozhi_light_create(
	const struct xiaozhi_light_config *config);

void xiaozhi_light_destroy(struct xiaozhi_light **light);

int xiaozhi_light_set(struct xiaozhi_light *light, int on);

int xiaozhi_light_get(struct xiaozhi_light *light, int *on);

#endif
