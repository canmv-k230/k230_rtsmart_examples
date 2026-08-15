#include "audio_file.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define AUDIO_FILE_MAX_SUFFIX 9999U

static k_s32 build_candidate(const char *requested_path, k_u32 index,
                             char *candidate, size_t candidate_size)
{
    const char *slash;
    const char *extension;
    size_t prefix_length;
    int length;

    if (index == 0)
    {
        length = snprintf(candidate, candidate_size, "%s", requested_path);
    }
    else
    {
        slash = strrchr(requested_path, '/');
        extension = strrchr(requested_path, '.');
        if (extension == NULL ||
            (slash != NULL && extension < slash + 1))
        {
            extension = requested_path + strlen(requested_path);
        }
        prefix_length = extension - requested_path;
        length = snprintf(candidate, candidate_size, "%.*s_%03u%s",
                          (int)prefix_length, requested_path, index, extension);
    }

    return length >= 0 && (size_t)length < candidate_size
               ? K_SUCCESS
               : K_FAILED;
}

static k_s32 split_path(const char *path, char *directory,
                        size_t directory_size, const char **filename)
{
    const char *slash;
    size_t length;

    if (path == NULL || path[0] != '/' || directory == NULL ||
        directory_size == 0 || filename == NULL)
    {
        return K_FAILED;
    }
    slash = strrchr(path, '/');
    if (slash == NULL || slash[1] == '\0')
    {
        return K_FAILED;
    }
    length = slash == path ? 1U : (size_t)(slash - path);
    if (length >= directory_size)
    {
        return K_FAILED;
    }
    memcpy(directory, path, length);
    directory[length] = '\0';
    *filename = slash + 1;
    return K_SUCCESS;
}

static k_bool parse_candidate_index(const char *name, const char *filename,
                                    k_u32 *index)
{
    const char *extension = strrchr(filename, '.');
    size_t stem_length;
    size_t extension_length;
    const char *digits;
    char *end;
    unsigned long value;

    if (strcmp(name, filename) == 0)
    {
        *index = 0;
        return K_TRUE;
    }
    if (extension == NULL)
    {
        extension = filename + strlen(filename);
    }
    stem_length = (size_t)(extension - filename);
    extension_length = strlen(extension);
    if (strncmp(name, filename, stem_length) != 0 || name[stem_length] != '_' ||
        strlen(name) <= stem_length + 1U + extension_length ||
        strcmp(name + strlen(name) - extension_length, extension) != 0)
    {
        return K_FALSE;
    }

    digits = name + stem_length + 1U;
    errno = 0;
    value = strtoul(digits, &end, 10);
    if (errno != 0 || end == digits || value == 0 ||
        value > AUDIO_FILE_MAX_SUFFIX ||
        (size_t)(end - digits) < 3U || strcmp(end, extension) != 0)
    {
        return K_FALSE;
    }
    *index = (k_u32)value;
    return K_TRUE;
}

k_s32 audio_file_open_unique(const char *requested_path, const char *mode,
                             FILE **file, char *actual_path,
                             size_t actual_path_size)
{
    char candidate[AUDIO_FILE_PATH_SIZE];
    int access_mode;

    if (requested_path == NULL || requested_path[0] != '/' || mode == NULL ||
        file == NULL || actual_path == NULL || actual_path_size == 0)
    {
        return K_FAILED;
    }
    if (strcmp(mode, "wb") == 0)
    {
        access_mode = O_WRONLY;
    }
    else if (strcmp(mode, "wb+") == 0)
    {
        access_mode = O_RDWR;
    }
    else
    {
        return K_FAILED;
    }

    *file = NULL;
    actual_path[0] = '\0';
    for (k_u32 index = 0; index <= AUDIO_FILE_MAX_SUFFIX; ++index)
    {
        int descriptor;
        FILE *stream;

        if (build_candidate(requested_path, index, candidate,
                            sizeof(candidate)) != K_SUCCESS)
        {
            return K_FAILED;
        }
        descriptor = open(candidate, access_mode | O_CREAT | O_EXCL, 0666);
        if (descriptor < 0)
        {
            if (errno == EEXIST)
            {
                continue;
            }
            return K_FAILED;
        }

        stream = fdopen(descriptor, mode);
        if (stream == NULL)
        {
            close(descriptor);
            remove(candidate);
            return K_FAILED;
        }
        if (snprintf(actual_path, actual_path_size, "%s", candidate) >=
            (int)actual_path_size)
        {
            fclose(stream);
            remove(candidate);
            return K_FAILED;
        }
        *file = stream;
        return K_SUCCESS;
    }

    return K_FAILED;
}

k_s32 audio_file_find_latest(const char *requested_path, char *actual_path,
                             size_t actual_path_size)
{
    char directory[AUDIO_FILE_PATH_SIZE];
    char candidate[AUDIO_FILE_PATH_SIZE];
    const char *filename;
    const char *separator;
    struct dirent *entry;
    struct stat status;
    time_t latest_time = 0;
    k_u32 latest_index = 0;
    k_bool found = K_FALSE;
    DIR *stream;

    if (actual_path == NULL || actual_path_size == 0 ||
        split_path(requested_path, directory, sizeof(directory), &filename) !=
            K_SUCCESS)
    {
        return K_FAILED;
    }
    actual_path[0] = '\0';
    stream = opendir(directory);
    if (stream == NULL)
    {
        return K_FAILED;
    }
    separator = strcmp(directory, "/") == 0 ? "" : "/";

    while ((entry = readdir(stream)) != NULL)
    {
        k_u32 index;

        if (!parse_candidate_index(entry->d_name, filename, &index) ||
            snprintf(candidate, sizeof(candidate), "%s%s%s", directory,
                     separator, entry->d_name) >= (int)sizeof(candidate) ||
            stat(candidate, &status) != 0 || !S_ISREG(status.st_mode))
        {
            continue;
        }
        if (!found || status.st_mtime > latest_time ||
            (status.st_mtime == latest_time && index > latest_index))
        {
            if (snprintf(actual_path, actual_path_size, "%s", candidate) >=
                (int)actual_path_size)
            {
                closedir(stream);
                actual_path[0] = '\0';
                return K_FAILED;
            }
            latest_time = status.st_mtime;
            latest_index = index;
            found = K_TRUE;
        }
    }
    if (closedir(stream) != 0)
    {
        return K_FAILED;
    }
    return found ? K_SUCCESS : K_FAILED;
}

k_s32 audio_file_discard(FILE **file, const char *path)
{
    k_s32 ret = K_SUCCESS;

    if (file != NULL && *file != NULL)
    {
        if (fclose(*file) != 0)
        {
            ret = K_FAILED;
        }
        *file = NULL;
    }
    if (path != NULL && path[0] != '\0' && remove(path) != 0 &&
        errno != ENOENT)
    {
        ret = K_FAILED;
    }
    return ret;
}
