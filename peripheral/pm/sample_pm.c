/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "mpi_pm_api.h"

#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define MAX_PROFILE_COUNT K_PM_MAX_PROFILE_COUNT
#define STRESS_DELAY_US 500000U

struct name_map {
    const char *name;
    int value;
};

static const struct name_map domain_map[] = {
    {"cpu", PM_DOMAIN_CPU},
    {"kpu", PM_DOMAIN_KPU},
    {"dpu", PM_DOMAIN_DPU},
    {"vpu", PM_DOMAIN_VPU},
    {"display", PM_DOMAIN_DISPLAY},
    {"disp", PM_DOMAIN_DISPLAY},
    {"media", PM_DOMAIN_MEDIA},
};

static const struct name_map governor_map[] = {
    {"manual", PM_GOVERNOR_MANUAL},
    {"performance", PM_GOVERNOR_PERFORMANCE},
    {"energysaving", PM_GOVERNOR_ENERGYSAVING},
    {"energy_saving", PM_GOVERNOR_ENERGYSAVING},
    {"energy-saving", PM_GOVERNOR_ENERGYSAVING},
    {"auto", PM_GOVERNOR_AUTO},
};

static const char *domain_names[] = {
    [PM_DOMAIN_CPU] = "cpu",
    [PM_DOMAIN_KPU] = "kpu",
    [PM_DOMAIN_DPU] = "dpu",
    [PM_DOMAIN_VPU] = "vpu",
    [PM_DOMAIN_DISPLAY] = "display",
    [PM_DOMAIN_MEDIA] = "media",
};

static const char *governor_names[] = {
    [PM_GOVERNOR_MANUAL] = "manual",
    [PM_GOVERNOR_PERFORMANCE] = "performance",
    [PM_GOVERNOR_ENERGYSAVING] = "energysaving",
    [PM_GOVERNOR_AUTO] = "auto",
};

static const char *prog_name = "pm";

static bool parse_long_value(const char *text, long *value)
{
    char *end = NULL;

    if (text == NULL || text[0] == '\0')
        return false;

    errno = 0;
    *value = strtol(text, &end, 0);
    return errno == 0 && end != text && *end == '\0';
}

static bool parse_int32(const char *text, int32_t *value)
{
    long temp;

    if (!parse_long_value(text, &temp) || temp < INT32_MIN || temp > INT32_MAX)
        return false;

    *value = (int32_t)temp;
    return true;
}

static bool parse_u32(const char *text, uint32_t *value)
{
    long temp;

    if (!parse_long_value(text, &temp) || temp < 0 ||
        (unsigned long)temp > UINT32_MAX)
        return false;

    *value = (uint32_t)temp;
    return true;
}

static bool parse_named_value(const char *text, const struct name_map *map,
    size_t map_count, int min, int max, int *value)
{
    long numeric;
    size_t i;

    if (parse_long_value(text, &numeric)) {
        if (numeric < min || numeric > max)
            return false;
        *value = (int)numeric;
        return true;
    }

    for (i = 0; i < map_count; i++) {
        if (strcasecmp(text, map[i].name) == 0) {
            *value = map[i].value;
            return true;
        }
    }

    return false;
}

static bool parse_domain(const char *text, k_pm_domain *domain)
{
    int value;

    if (!parse_named_value(text, domain_map, ARRAY_SIZE(domain_map), 0,
        PM_DOMAIN_NR - 1, &value))
        return false;

    *domain = (k_pm_domain)value;
    return true;
}

static bool parse_governor(const char *text, k_pm_governor *governor)
{
    int value;

    if (!parse_named_value(text, governor_map, ARRAY_SIZE(governor_map), 0,
        PM_GOVERNOR_AUTO, &value))
        return false;

    *governor = (k_pm_governor)value;
    return true;
}

static bool parse_on_off(const char *text, bool *enable)
{
    long numeric;

    if (parse_long_value(text, &numeric)) {
        if (numeric != 0 && numeric != 1)
            return false;
        *enable = numeric != 0;
        return true;
    }

    if (strcasecmp(text, "on") == 0 || strcasecmp(text, "enable") == 0 ||
        strcasecmp(text, "enabled") == 0 || strcasecmp(text, "true") == 0) {
        *enable = true;
        return true;
    }

    if (strcasecmp(text, "off") == 0 || strcasecmp(text, "disable") == 0 ||
        strcasecmp(text, "disabled") == 0 || strcasecmp(text, "false") == 0) {
        *enable = false;
        return true;
    }

    return false;
}

static const char *domain_name(k_pm_domain domain)
{
    int value = (int)domain;

    if (value >= 0 && value < (int)ARRAY_SIZE(domain_names) &&
        domain_names[value] != NULL)
        return domain_names[value];
    return "unknown";
}

static const char *governor_name(k_pm_governor governor)
{
    int value = (int)governor;

    if (value >= 0 && value < (int)ARRAY_SIZE(governor_names) &&
        governor_names[value] != NULL)
        return governor_names[value];
    return "unknown";
}

static int pm_error(const char *api, int ret)
{
    printf("%s failed: %#x\n", api, (unsigned int)ret);
    return 1;
}

static int usage_error(const char *usage)
{
    printf("Usage: %s %s\n", prog_name, usage);
    return 1;
}

static void show_help(void)
{
    printf("Usage:\n");
    printf("  %s help\n", prog_name);
    printf("  %s domains\n", prog_name);
    printf("  %s governors\n", prog_name);
    printf("  %s freq <domain>\n", prog_name);
    printf("  %s profiles <domain> [count]\n", prog_name);
    printf("  %s profile get <domain>\n", prog_name);
    printf("  %s profile set <domain> <profile>\n", prog_name);
    printf("  %s profile sweep <domain> [cycles]\n", prog_name);
    printf("  %s profile lock-sweep <domain> [cycles]\n", prog_name);
    printf("  %s governor get <domain>\n", prog_name);
    printf("  %s governor set <domain> <governor>\n", prog_name);
    printf("  %s lock <domain> <profile>\n", prog_name);
    printf("  %s unlock <domain> <profile>\n", prog_name);
    printf("  %s thermal get <domain>\n", prog_name);
    printf("  %s thermal protect <domain> <temp_c> <profile>\n", prog_name);
    printf("  %s thermal shutdown [temp_c]\n", prog_name);
    printf("  %s clock <domain> <on|off>\n", prog_name);
    printf("  %s power <domain> <on|off>\n", prog_name);
    printf("  %s stress <profile|lock|governor|power> <domain> <seconds>\n",
        prog_name);
    printf("  %s load <seconds>\n", prog_name);
    printf("\nThermal temperatures are integer Celsius in range -40 to 125.\n");
    printf("Examples:\n");
    printf("  %s freq cpu\n", prog_name);
    printf("  %s profiles cpu\n", prog_name);
    printf("  %s governor set cpu auto\n", prog_name);
    printf("  %s thermal protect cpu 85 -1\n", prog_name);
    printf("  %s thermal shutdown 100\n", prog_name);
}

static int cmd_help(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    show_help();
    return 0;
}

static int cmd_domains(int argc, char **argv)
{
    int i;

    (void)argv;
    if (argc != 0)
        return usage_error("domains");

    printf("Domains:\n");
    for (i = 0; i < PM_DOMAIN_NR; i++) {
        uint32_t count = 0;

        if (kd_mpi_pm_get_profiles((k_pm_domain)i, &count, NULL) == 0)
            printf("  %d: %s\n", i, domain_name((k_pm_domain)i));
    }

    return 0;
}

static int cmd_governors(int argc, char **argv)
{
    int i;

    (void)argv;
    if (argc != 0)
        return usage_error("governors");

    printf("Governors:\n");
    for (i = 0; i <= PM_GOVERNOR_AUTO; i++)
        printf("  %d: %s\n", i, governor_name((k_pm_governor)i));

    return 0;
}

static int cmd_profiles(int argc, char **argv)
{
    k_pm_domain domain;
    k_pm_profile *profiles = NULL;
    uint32_t count = 0;
    uint32_t i;
    int ret;

    if (argc < 1 || argc > 2)
        return usage_error("profiles <domain> [count]");
    if (!parse_domain(argv[0], &domain)) {
        printf("Invalid domain: %s\n", argv[0]);
        return 1;
    }

    if (argc == 2) {
        if (!parse_u32(argv[1], &count)) {
            printf("Invalid count: %s\n", argv[1]);
            return 1;
        }
        if (count > MAX_PROFILE_COUNT)
            count = MAX_PROFILE_COUNT;
    } else {
        ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
        if (ret)
            return pm_error("kd_mpi_pm_get_profiles", ret);
        if (count > MAX_PROFILE_COUNT)
            count = MAX_PROFILE_COUNT;
    }

    if (count == 0) {
        ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
        if (ret)
            return pm_error("kd_mpi_pm_get_profiles", ret);
        printf("%s profile count: %u\n", domain_name(domain), count);
        return 0;
    }

    profiles = malloc(sizeof(*profiles) * count);
    if (profiles == NULL) {
        printf("Out of memory\n");
        return 1;
    }

    ret = kd_mpi_pm_get_profiles(domain, &count, profiles);
    if (ret) {
        free(profiles);
        return pm_error("kd_mpi_pm_get_profiles", ret);
    }

    printf("%s profiles:\n", domain_name(domain));
    for (i = 0; i < count; i++)
        printf("  %2u: freq=%10d Hz volt=%7d uV\n", i,
            profiles[i].freq, profiles[i].volt);

    free(profiles);
    return 0;
}

static int profile_set_and_check(k_pm_domain domain, int32_t expected,
    uint32_t cycle, const char *direction)
{
    int32_t actual;
    int ret;

    ret = kd_mpi_pm_set_profile(domain, expected);
    if (ret) {
        printf("%s cycle %u %s set profile %d failed: %#x\n",
            domain_name(domain), cycle, direction, expected,
            (unsigned int)ret);
        return 1;
    }

    ret = kd_mpi_pm_get_profile(domain, &actual);
    if (ret) {
        printf("%s cycle %u %s read profile %d failed: %#x\n",
            domain_name(domain), cycle, direction, expected,
            (unsigned int)ret);
        return 1;
    }
    if (actual != expected) {
        printf("%s cycle %u %s profile mismatch: expected %d, got %d\n",
            domain_name(domain), cycle, direction, expected, actual);
        return 1;
    }

    return 0;
}

static int profile_sweep(k_pm_domain domain, uint32_t cycles)
{
    k_pm_profile profiles[MAX_PROFILE_COUNT];
    uint32_t order[MAX_PROFILE_COUNT];
    uint32_t count = 0;
    uint32_t cycle;
    uint32_t i;
    int32_t initial;
    int result = 1;
    int ret;

    ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
    if (ret)
        return pm_error("kd_mpi_pm_get_profiles", ret);
    if (count == 0 || count > MAX_PROFILE_COUNT) {
        printf("%s has unsupported profile count %u\n", domain_name(domain),
            count);
        return 1;
    }

    ret = kd_mpi_pm_get_profiles(domain, &count, profiles);
    if (ret)
        return pm_error("kd_mpi_pm_get_profiles", ret);
    ret = kd_mpi_pm_get_profile(domain, &initial);
    if (ret)
        return pm_error("kd_mpi_pm_get_profile", ret);

    for (i = 0; i < count; i++) {
        uint32_t j = i;

        order[i] = i;
        while (j > 0 && profiles[order[j - 1]].freq > profiles[order[j]].freq) {
            uint32_t temp = order[j - 1];

            order[j - 1] = order[j];
            order[j] = temp;
            j--;
        }
    }

    for (cycle = 1; cycle <= cycles; cycle++) {
        for (i = 0; i < count; i++) {
            if (profile_set_and_check(domain, (int32_t)order[i], cycle,
                "low-to-high"))
                goto restore;
        }
        for (i = count; i > 0; i--) {
            if (profile_set_and_check(domain, (int32_t)order[i - 1], cycle,
                "high-to-low"))
                goto restore;
        }
    }

    result = 0;

restore:
    ret = kd_mpi_pm_set_profile(domain, initial);
    if (ret) {
        printf("%s restore profile %d failed: %#x\n", domain_name(domain),
            initial, (unsigned int)ret);
        result = 1;
    } else {
        int32_t restored;

        ret = kd_mpi_pm_get_profile(domain, &restored);
        if (ret || restored != initial) {
            printf("%s restore mismatch: expected %d, got %d (ret=%#x)\n",
                domain_name(domain), initial, ret ? -1 : restored,
                (unsigned int)ret);
            result = 1;
        }
    }

    if (result == 0)
        printf("%s profile sweep PASS: %u cycles, %u profiles, %u transitions\n",
            domain_name(domain), cycles, count, cycles * count * 2);
    return result;
}

static int profile_read_and_check(k_pm_domain domain, int32_t expected,
    uint32_t cycle, const char *step)
{
    int32_t actual;
    int ret;

    ret = kd_mpi_pm_get_profile(domain, &actual);
    if (ret) {
        printf("%s cycle %u %s read profile failed: %#x\n",
            domain_name(domain), cycle, step, (unsigned int)ret);
        return 1;
    }
    if (actual != expected) {
        printf("%s cycle %u %s profile mismatch: expected %d, got %d\n",
            domain_name(domain), cycle, step, expected, actual);
        return 1;
    }

    return 0;
}

static int profile_lock_change(k_pm_domain domain, int32_t index, bool lock,
    uint32_t cycle, const char *step)
{
    int ret;

    ret = lock ? kd_mpi_pm_set_profile_lock(domain, index) :
        kd_mpi_pm_set_profile_unlock(domain, index);
    if (ret) {
        printf("%s cycle %u %s %s profile %d failed: %#x\n",
            domain_name(domain), cycle, step, lock ? "lock" : "unlock",
            index, (unsigned int)ret);
        return 1;
    }

    return 0;
}

static int profile_lock_sweep(k_pm_domain domain, uint32_t cycles)
{
    uint32_t count = 0;
    uint32_t cycle;
    int32_t initial;
    int32_t high;
    int32_t low;
    bool high_locked = false;
    bool low_locked = false;
    int result = 1;
    int ret;

    ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
    if (ret)
        return pm_error("kd_mpi_pm_get_profiles", ret);
    if (count < 2 || count > MAX_PROFILE_COUNT) {
        printf("%s needs at least two profiles, got %u\n",
            domain_name(domain), count);
        return 1;
    }

    ret = kd_mpi_pm_get_profile(domain, &initial);
    if (ret)
        return pm_error("kd_mpi_pm_get_profile", ret);

    high = 0;
    low = (int32_t)count - 1;
    for (cycle = 1; cycle <= cycles; cycle++) {
        if (profile_lock_change(domain, low, true, cycle,
            "reverse first"))
            goto cleanup;
        low_locked = true;
        if (profile_read_and_check(domain, low, cycle,
            "reverse first vote"))
            goto cleanup;

        if (profile_lock_change(domain, high, true, cycle,
            "reverse second"))
            goto cleanup;
        high_locked = true;
        if (profile_read_and_check(domain, high, cycle,
            "reverse winning vote"))
            goto cleanup;

        if (profile_lock_change(domain, high, false, cycle,
            "reverse first unlock"))
            goto cleanup;
        high_locked = false;
        if (profile_read_and_check(domain, low, cycle,
            "reverse remaining vote"))
            goto cleanup;

        if (profile_lock_change(domain, low, false, cycle,
            "reverse final unlock"))
            goto cleanup;
        low_locked = false;
        if (profile_read_and_check(domain, initial, cycle,
            "reverse restore"))
            goto cleanup;

        if (profile_lock_change(domain, low, true, cycle,
            "forward first"))
            goto cleanup;
        low_locked = true;
        if (profile_lock_change(domain, high, true, cycle,
            "forward second"))
            goto cleanup;
        high_locked = true;
        if (profile_read_and_check(domain, high, cycle,
            "forward winning vote"))
            goto cleanup;

        if (profile_lock_change(domain, low, false, cycle,
            "forward first unlock"))
            goto cleanup;
        low_locked = false;
        if (profile_read_and_check(domain, high, cycle,
            "forward remaining vote"))
            goto cleanup;

        if (profile_lock_change(domain, high, false, cycle,
            "forward final unlock"))
            goto cleanup;
        high_locked = false;
        if (profile_read_and_check(domain, initial, cycle,
            "forward restore"))
            goto cleanup;
    }

    ret = kd_mpi_pm_set_profile_unlock(domain, high);
    if (ret == 0) {
        printf("%s accepted an unmatched profile %d unlock\n",
            domain_name(domain), high);
        goto cleanup;
    }

    result = 0;

cleanup:
    if (high_locked && kd_mpi_pm_set_profile_unlock(domain, high))
        result = 1;
    if (low_locked && kd_mpi_pm_set_profile_unlock(domain, low))
        result = 1;
    ret = kd_mpi_pm_set_profile(domain, initial);
    if (ret || profile_read_and_check(domain, initial, cycle, "cleanup"))
        result = 1;

    if (result == 0)
        printf("%s profile lock sweep PASS: %u cycles, %u lock transitions\n",
            domain_name(domain), cycles, cycles * 8);
    return result;
}

static int cmd_profile(int argc, char **argv)
{
    k_pm_domain domain;
    int32_t index;
    int ret;

    if (argc < 2)
        return usage_error("profile get <domain> | profile set <domain> <profile>");
    if (!parse_domain(argv[1], &domain)) {
        printf("Invalid domain: %s\n", argv[1]);
        return 1;
    }

    if (strcasecmp(argv[0], "get") == 0) {
        if (argc != 2)
            return usage_error("profile get <domain>");
        ret = kd_mpi_pm_get_profile(domain, &index);
        if (ret)
            return pm_error("kd_mpi_pm_get_profile", ret);
        printf("%s profile: %d\n", domain_name(domain), index);
        return 0;
    }

    if (strcasecmp(argv[0], "set") == 0) {
        if (argc != 3)
            return usage_error("profile set <domain> <profile>");
        if (!parse_int32(argv[2], &index)) {
            printf("Invalid profile: %s\n", argv[2]);
            return 1;
        }
        ret = kd_mpi_pm_set_profile(domain, index);
        if (ret)
            return pm_error("kd_mpi_pm_set_profile", ret);
        printf("%s set profile %d\n", domain_name(domain), index);
        return 0;
    }

    if (strcasecmp(argv[0], "sweep") == 0) {
        uint32_t cycles = 1;

        if (argc < 2 || argc > 3)
            return usage_error("profile sweep <domain> [cycles]");
        if (argc == 3 && (!parse_u32(argv[2], &cycles) || cycles == 0)) {
            printf("Invalid cycle count: %s\n", argv[2]);
            return 1;
        }
        return profile_sweep(domain, cycles);
    }

    if (strcasecmp(argv[0], "lock-sweep") == 0) {
        uint32_t cycles = 1;

        if (argc < 2 || argc > 3)
            return usage_error("profile lock-sweep <domain> [cycles]");
        if (argc == 3 && (!parse_u32(argv[2], &cycles) || cycles == 0)) {
            printf("Invalid cycle count: %s\n", argv[2]);
            return 1;
        }
        return profile_lock_sweep(domain, cycles);
    }

    return usage_error("profile get <domain> | profile set <domain> <profile> | profile sweep <domain> [cycles] | profile lock-sweep <domain> [cycles]");
}

static int cmd_freq(int argc, char **argv)
{
    k_pm_domain domain;
    k_pm_profile *profiles;
    uint32_t count = 0;
    uint32_t profile;
    int32_t index;
    int ret;

    if (argc != 1)
        return usage_error("freq <domain>");
    if (!parse_domain(argv[0], &domain)) {
        printf("Invalid domain: %s\n", argv[0]);
        return 1;
    }

    ret = kd_mpi_pm_get_profile(domain, &index);
    if (ret)
        return pm_error("kd_mpi_pm_get_profile", ret);
    if (index < 0) {
        printf("%s current profile is invalid: %d\n", domain_name(domain),
            index);
        return 1;
    }

    profile = (uint32_t)index;
    ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
    if (ret)
        return pm_error("kd_mpi_pm_get_profiles", ret);
    if (count == 0 || profile >= count) {
        printf("%s current profile %u is outside profile count %u\n",
            domain_name(domain), profile, count);
        return 1;
    }
    if (count > MAX_PROFILE_COUNT)
        count = MAX_PROFILE_COUNT;
    if (profile >= count) {
        printf("%s current profile %u is outside sample read limit %u\n",
            domain_name(domain), profile, count);
        return 1;
    }

    profiles = malloc(sizeof(*profiles) * count);
    if (profiles == NULL) {
        printf("Out of memory\n");
        return 1;
    }

    ret = kd_mpi_pm_get_profiles(domain, &count, profiles);
    if (ret) {
        free(profiles);
        return pm_error("kd_mpi_pm_get_profiles", ret);
    }
    if (profile >= count) {
        printf("%s current profile %u is outside returned profile count %u\n",
            domain_name(domain), profile, count);
        free(profiles);
        return 1;
    }

    printf("%s profile %u frequency: %d Hz\n", domain_name(domain),
        profile, profiles[profile].freq);
    free(profiles);
    return 0;
}

static int cmd_governor(int argc, char **argv)
{
    k_pm_domain domain;
    k_pm_governor governor;
    int ret;

    if (argc < 2)
        return usage_error("governor get <domain> | governor set <domain> <governor>");
    if (!parse_domain(argv[1], &domain)) {
        printf("Invalid domain: %s\n", argv[1]);
        return 1;
    }

    if (strcasecmp(argv[0], "get") == 0) {
        if (argc != 2)
            return usage_error("governor get <domain>");
        ret = kd_mpi_pm_get_governor(domain, &governor);
        if (ret)
            return pm_error("kd_mpi_pm_get_governor", ret);
        printf("%s governor: %s (%d)\n", domain_name(domain),
            governor_name(governor), governor);
        return 0;
    }

    if (strcasecmp(argv[0], "set") == 0) {
        if (argc != 3)
            return usage_error("governor set <domain> <governor>");
        if (!parse_governor(argv[2], &governor)) {
            printf("Invalid governor: %s\n", argv[2]);
            return 1;
        }
        ret = kd_mpi_pm_set_governor(domain, governor);
        if (ret)
            return pm_error("kd_mpi_pm_set_governor", ret);
        printf("%s set governor %s\n", domain_name(domain),
            governor_name(governor));
        return 0;
    }

    return usage_error("governor get <domain> | governor set <domain> <governor>");
}

static int set_profile_lock_state(const char *api, bool lock, int argc,
    char **argv)
{
    k_pm_domain domain;
    int32_t index;
    int ret;

    if (argc != 2)
        return usage_error(lock ? "lock <domain> <profile>" :
            "unlock <domain> <profile>");
    if (!parse_domain(argv[0], &domain)) {
        printf("Invalid domain: %s\n", argv[0]);
        return 1;
    }
    if (!parse_int32(argv[1], &index)) {
        printf("Invalid profile: %s\n", argv[1]);
        return 1;
    }

    ret = lock ? kd_mpi_pm_set_profile_lock(domain, index) :
        kd_mpi_pm_set_profile_unlock(domain, index);
    if (ret)
        return pm_error(api, ret);

    printf("%s %s profile %d\n", domain_name(domain),
        lock ? "locked" : "unlocked", index);
    return 0;
}

static int cmd_lock(int argc, char **argv)
{
    return set_profile_lock_state("kd_mpi_pm_set_profile_lock", true, argc,
        argv);
}

static int cmd_unlock(int argc, char **argv)
{
    return set_profile_lock_state("kd_mpi_pm_set_profile_unlock", false, argc,
        argv);
}

static int cmd_thermal(int argc, char **argv)
{
    k_pm_domain domain;
    int32_t temp;
    int32_t index;
    int ret;

    if (argc < 1)
        return usage_error("thermal get <domain> | thermal protect <domain> <temp_c> <profile> | thermal shutdown [temp_c]");

    if (strcasecmp(argv[0], "get") == 0) {
        if (argc != 2)
            return usage_error("thermal get <domain>");
        if (!parse_domain(argv[1], &domain)) {
            printf("Invalid domain: %s\n", argv[1]);
            return 1;
        }
        ret = kd_mpi_pm_get_thermal_protect(domain, &temp, &index);
        if (ret)
            return pm_error("kd_mpi_pm_get_thermal_protect", ret);
        printf("%s thermal protect: %d C limit profile %d\n",
            domain_name(domain), temp, index);
        return 0;
    }

    if (strcasecmp(argv[0], "protect") == 0) {
        if (argc != 4)
            return usage_error("thermal protect <domain> <temp_c> <profile>");
        if (!parse_domain(argv[1], &domain)) {
            printf("Invalid domain: %s\n", argv[1]);
            return 1;
        }
        if (!parse_int32(argv[2], &temp)) {
            printf("Invalid temperature: %s\n", argv[2]);
            return 1;
        }
        if (!parse_int32(argv[3], &index)) {
            printf("Invalid profile: %s\n", argv[3]);
            return 1;
        }
        ret = kd_mpi_pm_set_thermal_protect(domain, temp, index);
        if (ret)
            return pm_error("kd_mpi_pm_set_thermal_protect", ret);
        printf("%s set thermal protect %d C limit profile %d\n",
            domain_name(domain), temp, index);
        return 0;
    }

    if (strcasecmp(argv[0], "shutdown") == 0) {
        if (argc == 1) {
            ret = kd_mpi_pm_get_thermal_shutdown(&temp);
            if (ret)
                return pm_error("kd_mpi_pm_get_thermal_shutdown", ret);
            printf("chip thermal shutdown: %d C\n", temp);
            return 0;
        }
        if (argc != 2)
            return usage_error("thermal shutdown [temp_c]");
        if (!parse_int32(argv[1], &temp)) {
            printf("Invalid temperature: %s\n", argv[1]);
            return 1;
        }
        ret = kd_mpi_pm_set_thermal_shutdown(temp);
        if (ret)
            return pm_error("kd_mpi_pm_set_thermal_shutdown", ret);
        printf("chip set thermal shutdown %d C\n", temp);
        return 0;
    }

    return usage_error("thermal get <domain> | thermal protect <domain> <temp_c> <profile> | thermal shutdown [temp_c]");
}

static int set_domain_switch(const char *api, bool is_power, int argc,
    char **argv)
{
    k_pm_domain domain;
    bool enable;
    int ret;

    if (argc != 2)
        return usage_error(is_power ? "power <domain> <on|off>" :
            "clock <domain> <on|off>");
    if (!parse_domain(argv[0], &domain)) {
        printf("Invalid domain: %s\n", argv[0]);
        return 1;
    }
    if (!parse_on_off(argv[1], &enable)) {
        printf("Invalid switch value: %s\n", argv[1]);
        return 1;
    }

    ret = is_power ? kd_mpi_pm_set_power(domain, enable) :
        kd_mpi_pm_set_clock(domain, enable);
    if (ret)
        return pm_error(api, ret);

    printf("%s %s %s\n", domain_name(domain), enable ? "enabled" : "disabled",
        is_power ? "power" : "clock");
    return 0;
}

static int cmd_clock(int argc, char **argv)
{
    return set_domain_switch("kd_mpi_pm_set_clock", false, argc, argv);
}

static int cmd_power(int argc, char **argv)
{
    return set_domain_switch("kd_mpi_pm_set_power", true, argc, argv);
}

static int stress_deadline(uint32_t seconds, uint64_t *deadline_ms)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now)) {
        printf("clock_gettime failed: %d\n", errno);
        return 1;
    }

    *deadline_ms = (uint64_t)now.tv_sec * 1000U +
        (uint64_t)now.tv_nsec / 1000000U + (uint64_t)seconds * 1000U;
    return 0;
}

static int stress_before_deadline(uint64_t deadline_ms, bool *before)
{
    struct timespec now;
    uint64_t now_ms;

    if (clock_gettime(CLOCK_MONOTONIC, &now)) {
        printf("clock_gettime failed: %d\n", errno);
        return 1;
    }

    now_ms = (uint64_t)now.tv_sec * 1000U +
        (uint64_t)now.tv_nsec / 1000000U;
    *before = now_ms < deadline_ms;
    return 0;
}

static int stress_profile(k_pm_domain domain, uint32_t seconds)
{
    k_pm_governor initial_governor;
    uint64_t deadline_ms;
    uint32_t count = 0;
    uint32_t operations = 0;
    int32_t initial_profile;
    bool running;
    int result = 1;
    int ret;

    ret = kd_mpi_pm_get_governor(domain, &initial_governor);
    if (ret)
        return pm_error("kd_mpi_pm_get_governor", ret);
    if (initial_governor != PM_GOVERNOR_MANUAL) {
        ret = kd_mpi_pm_set_governor(domain, PM_GOVERNOR_MANUAL);
        if (ret)
            return pm_error("kd_mpi_pm_set_governor", ret);
    }

    ret = kd_mpi_pm_get_profile(domain, &initial_profile);
    if (ret)
        goto restore_governor;
    ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
    if (ret) {
        pm_error("kd_mpi_pm_get_profiles", ret);
        goto restore_profile;
    }
    if (count == 0 || count > MAX_PROFILE_COUNT) {
        printf("%s has unsupported profile count %u\n", domain_name(domain),
            count);
        goto restore_profile;
    }
    if (stress_deadline(seconds, &deadline_ms))
        goto restore_profile;

    for (;;) {
        if (stress_before_deadline(deadline_ms, &running))
            goto restore_profile;
        if (!running)
            break;
        if (profile_set_and_check(domain, (int32_t)(operations % count),
            operations + 1, "stress"))
            goto restore_profile;
        operations++;
        usleep(STRESS_DELAY_US);
    }
    result = 0;

restore_profile:
    ret = kd_mpi_pm_set_profile(domain, initial_profile);
    if (ret || profile_read_and_check(domain, initial_profile,
        operations + 1, "stress cleanup"))
        result = 1;
restore_governor:
    ret = kd_mpi_pm_set_governor(domain, initial_governor);
    if (ret) {
        printf("%s restore governor %s failed: %#x\n", domain_name(domain),
            governor_name(initial_governor), (unsigned int)ret);
        result = 1;
    }

    if (result == 0)
        printf("%s profile stress PASS: %u seconds, %u operations\n",
            domain_name(domain), seconds, operations);
    return result;
}

static int stress_lock(k_pm_domain domain, uint32_t seconds)
{
    k_pm_governor initial_governor;
    uint64_t deadline_ms;
    uint32_t count = 0;
    uint32_t cycles = 0;
    int32_t initial_profile;
    int32_t high;
    int32_t low;
    bool high_locked = false;
    bool low_locked = false;
    bool running;
    int result = 1;
    int ret;

    ret = kd_mpi_pm_get_governor(domain, &initial_governor);
    if (ret)
        return pm_error("kd_mpi_pm_get_governor", ret);
    if (initial_governor != PM_GOVERNOR_MANUAL) {
        ret = kd_mpi_pm_set_governor(domain, PM_GOVERNOR_MANUAL);
        if (ret)
            return pm_error("kd_mpi_pm_set_governor", ret);
    }

    ret = kd_mpi_pm_get_profile(domain, &initial_profile);
    if (ret)
        goto restore_governor;
    ret = kd_mpi_pm_get_profiles(domain, &count, NULL);
    if (ret) {
        pm_error("kd_mpi_pm_get_profiles", ret);
        goto restore_profile;
    }
    if (count < 2 || count > MAX_PROFILE_COUNT) {
        printf("%s needs at least two profiles, got %u\n",
            domain_name(domain), count);
        goto restore_profile;
    }
    if (stress_deadline(seconds, &deadline_ms))
        goto restore_profile;

    high = 0;
    low = (int32_t)count - 1;
    for (;;) {
        uint32_t cycle = cycles + 1;

        if (stress_before_deadline(deadline_ms, &running))
            goto cleanup_locks;
        if (!running)
            break;

        if (profile_lock_change(domain, low, true, cycle, "stress low"))
            goto cleanup_locks;
        low_locked = true;
        if (profile_read_and_check(domain, low, cycle, "stress low vote"))
            goto cleanup_locks;
        if (profile_lock_change(domain, high, true, cycle, "stress high"))
            goto cleanup_locks;
        high_locked = true;
        if (profile_read_and_check(domain, high, cycle, "stress high vote"))
            goto cleanup_locks;
        if (profile_lock_change(domain, low, false, cycle,
            "stress low unlock"))
            goto cleanup_locks;
        low_locked = false;
        if (profile_read_and_check(domain, high, cycle,
            "stress remaining high vote"))
            goto cleanup_locks;
        if (profile_lock_change(domain, high, false, cycle,
            "stress high unlock"))
            goto cleanup_locks;
        high_locked = false;
        if (profile_read_and_check(domain, initial_profile, cycle,
            "stress restore"))
            goto cleanup_locks;

        cycles++;
        usleep(STRESS_DELAY_US);
    }
    result = 0;

cleanup_locks:
    if (high_locked && kd_mpi_pm_set_profile_unlock(domain, high))
        result = 1;
    if (low_locked && kd_mpi_pm_set_profile_unlock(domain, low))
        result = 1;
restore_profile:
    ret = kd_mpi_pm_set_profile(domain, initial_profile);
    if (ret || profile_read_and_check(domain, initial_profile, cycles + 1,
        "stress cleanup"))
        result = 1;
restore_governor:
    ret = kd_mpi_pm_set_governor(domain, initial_governor);
    if (ret) {
        printf("%s restore governor %s failed: %#x\n", domain_name(domain),
            governor_name(initial_governor), (unsigned int)ret);
        result = 1;
    }

    if (result == 0)
        printf("%s lock stress PASS: %u seconds, %u cycles, %u operations\n",
            domain_name(domain), seconds, cycles, cycles * 4);
    return result;
}

static int stress_governor(k_pm_domain domain, uint32_t seconds)
{
    static const k_pm_governor governors[] = {
        PM_GOVERNOR_PERFORMANCE,
        PM_GOVERNOR_ENERGYSAVING,
        PM_GOVERNOR_MANUAL,
    };
    k_pm_governor initial_governor;
    uint64_t deadline_ms;
    uint32_t operations = 0;
    bool running;
    int result = 1;
    int ret;

    ret = kd_mpi_pm_get_governor(domain, &initial_governor);
    if (ret)
        return pm_error("kd_mpi_pm_get_governor", ret);
    if (stress_deadline(seconds, &deadline_ms))
        return 1;

    for (;;) {
        k_pm_governor actual;
        k_pm_governor target;

        if (stress_before_deadline(deadline_ms, &running))
            goto restore;
        if (!running)
            break;

        target = governors[operations % ARRAY_SIZE(governors)];
        ret = kd_mpi_pm_set_governor(domain, target);
        if (ret) {
            printf("%s governor stress set %s failed at operation %u: %#x\n",
                domain_name(domain), governor_name(target), operations + 1,
                (unsigned int)ret);
            goto restore;
        }
        ret = kd_mpi_pm_get_governor(domain, &actual);
        if (ret || actual != target) {
            printf("%s governor stress mismatch at operation %u: expected %s, got %s (ret=%#x)\n",
                domain_name(domain), operations + 1, governor_name(target),
                ret ? "unknown" : governor_name(actual), (unsigned int)ret);
            goto restore;
        }
        operations++;
        usleep(STRESS_DELAY_US);
    }
    result = 0;

restore:
    ret = kd_mpi_pm_set_governor(domain, initial_governor);
    if (ret) {
        printf("%s restore governor %s failed: %#x\n", domain_name(domain),
            governor_name(initial_governor), (unsigned int)ret);
        result = 1;
    } else {
        k_pm_governor restored;

        ret = kd_mpi_pm_get_governor(domain, &restored);
        if (ret || restored != initial_governor) {
            printf("%s restore governor mismatch: expected %s, got %s (ret=%#x)\n",
                domain_name(domain), governor_name(initial_governor),
                ret ? "unknown" : governor_name(restored),
                (unsigned int)ret);
            result = 1;
        }
    }

    if (result == 0)
        printf("%s governor stress PASS: %u seconds, %u operations\n",
            domain_name(domain), seconds, operations);
    return result;
}

static int stress_power(k_pm_domain domain, uint32_t seconds)
{
    uint64_t deadline_ms;
    uint32_t cycles = 0;
    bool clock_enabled = false;
    bool power_enabled = false;
    bool running;
    int result = 1;
    int ret;

    if (stress_deadline(seconds, &deadline_ms))
        return 1;

    for (;;) {
        if (stress_before_deadline(deadline_ms, &running))
            goto cleanup;
        if (!running)
            break;

        ret = kd_mpi_pm_set_clock(domain, true);
        if (ret) {
            printf("%s power stress clock on failed at cycle %u: %#x\n",
                domain_name(domain), cycles + 1, (unsigned int)ret);
            goto cleanup;
        }
        clock_enabled = true;
        ret = kd_mpi_pm_set_power(domain, true);
        if (ret) {
            printf("%s power stress power on failed at cycle %u: %#x\n",
                domain_name(domain), cycles + 1, (unsigned int)ret);
            goto cleanup;
        }
        power_enabled = true;
        ret = kd_mpi_pm_set_power(domain, false);
        if (ret) {
            printf("%s power stress power off failed at cycle %u: %#x\n",
                domain_name(domain), cycles + 1, (unsigned int)ret);
            goto cleanup;
        }
        power_enabled = false;
        ret = kd_mpi_pm_set_clock(domain, false);
        if (ret) {
            printf("%s power stress clock off failed at cycle %u: %#x\n",
                domain_name(domain), cycles + 1, (unsigned int)ret);
            goto cleanup;
        }
        clock_enabled = false;
        cycles++;
        usleep(STRESS_DELAY_US);
    }
    result = 0;

cleanup:
    if (power_enabled && kd_mpi_pm_set_power(domain, false))
        result = 1;
    if (clock_enabled && kd_mpi_pm_set_clock(domain, false))
        result = 1;

    if (result == 0)
        printf("%s power stress PASS: %u seconds, %u cycles, %u operations\n",
            domain_name(domain), seconds, cycles, cycles * 4);
    return result;
}

static int cmd_stress(int argc, char **argv)
{
    k_pm_domain domain;
    uint32_t seconds;

    if (argc != 3)
        return usage_error("stress <profile|lock|governor|power> <domain> <seconds>");
    if (!parse_domain(argv[1], &domain)) {
        printf("Invalid domain: %s\n", argv[1]);
        return 1;
    }
    if (!parse_u32(argv[2], &seconds) || seconds == 0) {
        printf("Invalid duration: %s\n", argv[2]);
        return 1;
    }

    if (strcasecmp(argv[0], "profile") == 0)
        return stress_profile(domain, seconds);
    if (strcasecmp(argv[0], "lock") == 0)
        return stress_lock(domain, seconds);
    if (strcasecmp(argv[0], "governor") == 0)
        return stress_governor(domain, seconds);
    if (strcasecmp(argv[0], "power") == 0)
        return stress_power(domain, seconds);

    return usage_error("stress <profile|lock|governor|power> <domain> <seconds>");
}

static int cmd_load(int argc, char **argv)
{
    volatile uint64_t accumulator = 1;
    uint64_t deadline_ms;
    uint64_t iterations = 0;
    uint32_t seconds;
    bool running;
    unsigned int i;

    if (argc != 1)
        return usage_error("load <seconds>");
    if (!parse_u32(argv[0], &seconds) || seconds == 0) {
        printf("Invalid duration: %s\n", argv[0]);
        return 1;
    }
    if (stress_deadline(seconds, &deadline_ms))
        return 1;

    for (;;) {
        if (stress_before_deadline(deadline_ms, &running))
            return 1;
        if (!running)
            break;

        for (i = 0; i < 100000U; i++)
            accumulator = accumulator * UINT64_C(6364136223846793005) + 1U;
        iterations += i;
    }

    printf("CPU load PASS: %u seconds, %llu iterations, checksum=%#llx\n",
        seconds, (unsigned long long)iterations,
        (unsigned long long)accumulator);
    return 0;
}

struct command {
    const char *name;
    int (*func)(int argc, char **argv);
};

static const struct command commands[] = {
    {"help", cmd_help},
    {"domains", cmd_domains},
    {"governors", cmd_governors},
    {"freq", cmd_freq},
    {"profiles", cmd_profiles},
    {"profile", cmd_profile},
    {"governor", cmd_governor},
    {"lock", cmd_lock},
    {"unlock", cmd_unlock},
    {"thermal", cmd_thermal},
    {"clock", cmd_clock},
    {"power", cmd_power},
    {"stress", cmd_stress},
    {"load", cmd_load},
};

int main(int argc, char **argv)
{
    size_t i;

    if (argc > 0 && argv[0] != NULL && argv[0][0] != '\0')
        prog_name = argv[0];

    if (argc < 2) {
        show_help();
        return 0;
    }

    for (i = 0; i < ARRAY_SIZE(commands); i++) {
        if (strcasecmp(argv[1], commands[i].name) == 0)
            return commands[i].func(argc - 2, &argv[2]);
    }

    printf("Unknown command: %s\n\n", argv[1]);
    show_help();
    return 1;
}
