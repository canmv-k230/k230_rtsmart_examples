#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string>

#include "minihttp.h"

class HttpDumpSocket : public minihttp::HttpSocket
{
public:
    virtual ~HttpDumpSocket() {}

protected:
    virtual void _OnClose()
    {
        puts("\n[Event] _OnClose()");
        minihttp::HttpSocket::_OnClose();
    }
    
    virtual void _OnOpen()
    {
        puts("[Event] _OnOpen()");
        char buf[1024] = { 0 };
        // Check SSL verification results
        minihttp::SSLResult sr = verifySSL(buf, sizeof(buf));
        printf("SSL status flags (0 is good): 0x%x\n", sr);
        if (sr != 0) {
            printf("SSL Warning/Error Info: %s\n", buf);
            printf("Note: If error is 0x1, check your system 'date'!\n");
        }
        minihttp::HttpSocket::_OnOpen();
    }

    virtual void _OnRequestDone()
    {
        printf("[Event] _OnRequestDone(): %s\n", GetCurrentRequest().resource.c_str());
    }

    virtual void _OnRecv(void *buf, unsigned int size)
    {
        if(!size) return;
        printf("\n--- Data Received (%u bytes) ---\n", size);
        fwrite(buf, 1, size, stdout);
        printf("\n-------------------------------\n");
    }
};

int main(int argc, char *argv[])
{
    // Check for user input
    if (argc < 2) {
        printf("Usage: %s <url>\n", argv[0]);
        printf("Example: %s https://www.baidu.com\n", argv[0]);
        return -1;
    }

    const char* target_url = argv[1];

#ifdef SIGPIPE
    signal(SIGPIPE, SIG_IGN);
#endif

    printf("minihttp have ssl support: %s\n", minihttp::HasSSL() ? "YES" : "NO");

    minihttp::InitNetwork();
    atexit(minihttp::StopNetwork);

    HttpDumpSocket *ht = new HttpDumpSocket;
    ht->SetKeepAlive(3); // Set to 0 for simple one-off tests
    ht->SetBufsizeIn(64 * 1024);

    printf("Connecting to: %s\n", target_url);

    // Pass the user input directly to Download
    if (!ht->Download(target_url)) {
        printf("Error: Invalid URL or failed to initiate download.\n");
        delete ht;
        return -1;
    }

    minihttp::SocketSet ss;
    ss.add(ht, true); // true: auto-delete socket when finished

    // Loop until the download task is complete
    while(ss.size()) {
        ss.update();
        // Short sleep can be added here if RT-Smart supports it to save CPU
    }

    printf("\nTest finished.\n");
    return 0;
}
