#include "xiaozhi_wake.h"

#include "kws.h"

#include <stdio.h>

#include <exception>
#include <new>
#include <string>
#include <string.h>
#include <vector>

struct xiaozhi_wake {
	KWS *model;
	size_t cooldown_samples;
	std::vector<int16_t> pending_samples;
	std::string model_path;
	std::string task_name;
	int keyword_count;
	float threshold;
};

static int load_model(struct xiaozhi_wake *wake)
{
	try {
		wake->model = new KWS(wake->model_path.c_str(), wake->task_name,
			wake->keyword_count, wake->threshold, 0);
	} catch (const std::exception &error) {
		printf("xiaozhi: wake-word model initialization failed: %s\n",
		       error.what());
		return -1;
	} catch (...) {
		printf("xiaozhi: wake-word model initialization failed\n");
		return -1;
	}
	return 0;
}

extern "C" struct xiaozhi_wake *xiaozhi_wake_create(
	const struct xiaozhi_wake_config *config)
{
	struct xiaozhi_wake *wake;
	FILE *model_file;

	if (!config || !config->model_path || !config->task_name ||
	    config->keyword_count <= 0 || config->threshold < 0.0f ||
	    config->threshold > 1.0f)
		return NULL;
	if ((!strcmp(config->task_name, "xiaonan") &&
	     config->keyword_count != 2) ||
	    (!strcmp(config->task_name, "xiaowen") &&
	     config->keyword_count != 3) ||
	    (!strcmp(config->task_name, "commands") &&
	     config->keyword_count != 11) ||
	    (strcmp(config->task_name, "xiaonan") &&
	     strcmp(config->task_name, "xiaowen") &&
	     strcmp(config->task_name, "commands")))
		return NULL;
	model_file = fopen(config->model_path, "rb");
	if (!model_file)
		return NULL;
	fclose(model_file);

	wake = new (std::nothrow) struct xiaozhi_wake();
	if (!wake)
		return NULL;
	wake->model = NULL;
	wake->cooldown_samples = 0;
	wake->model_path = config->model_path;
	wake->task_name = config->task_name;
	wake->keyword_count = config->keyword_count;
	wake->threshold = config->threshold;
	if (load_model(wake)) {
		delete wake;
		return NULL;
	}
	return wake;
}

extern "C" void xiaozhi_wake_destroy(struct xiaozhi_wake *wake)
{
	if (!wake)
		return;
	delete wake->model;
	delete wake;
}

extern "C" int xiaozhi_wake_reset(struct xiaozhi_wake *wake)
{
	if (!wake)
		return -1;
	delete wake->model;
	wake->model = NULL;
	wake->cooldown_samples = 0;
	wake->pending_samples.clear();
	KWS::feature_pipeline.Reset();
	KWS::feature_pipeline.AcceptWaveform(
		std::vector<float>(320, 0.0f));
	return load_model(wake);
}

extern "C" int xiaozhi_wake_process(struct xiaozhi_wake *wake,
					     const int16_t *samples, size_t sample_count)
{
	const size_t chunk_samples = XIAOZHI_WAKE_WORD_FRAME_SAMPLES;

	if (!wake || !wake->model || !samples || !sample_count)
		return -1;
	wake->pending_samples.insert(wake->pending_samples.end(), samples,
					     samples + sample_count);
	while (wake->pending_samples.size() >= chunk_samples) {
		std::vector<float> pcm(chunk_samples);
		std::string result;

		for (size_t i = 0; i < chunk_samples; i++)
			pcm[i] = (float)wake->pending_samples[i];
		wake->pending_samples.erase(
			wake->pending_samples.begin(),
			wake->pending_samples.begin() + chunk_samples);
		if (!wake->model->pre_process(pcm))
			continue;
		wake->model->inference();
		result = wake->model->post_process();
		if (wake->cooldown_samples > chunk_samples)
			wake->cooldown_samples -= chunk_samples;
		else
			wake->cooldown_samples = 0;
		if (!wake->cooldown_samples && result != "Deactivated!") {
			wake->cooldown_samples = 2 * 16000;
			return 1;
		}
	}
	return 0;
}
