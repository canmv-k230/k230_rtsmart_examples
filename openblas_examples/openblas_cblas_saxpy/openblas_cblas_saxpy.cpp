#include "cblas.h"
#include <iostream>
#include <string>
#include <sstream>
#define N 4

using namespace std;

string expect = R"(4 7 11 14 )";

/**
 * @brief 主函数，程序的入口点
 * 
 * 此函数用于测试 OpenBLAS 库中的 cblas_saxpy 函数。
 * 它将向量 x 乘以标量 alpha 后加到向量 y 上，
 * 并将结果与预期结果进行比较，输出测试结果。
 * 
 * @param argc 命令行参数的数量
 * @param argv 命令行参数的数组
 * @return int 程序退出状态码，0 表示正常退出
 */
int main(int argc, char ** argv)
{
    // 定义标量 alpha，用于向量 x 的缩放
    float alpha=3;
    // 定义向量 x，包含 4 个浮点数
    float x[4]={1.0,2,3,4};
    /* 向量 x 的元素为 1,2,3,4
     */
    // 定义向量 y，包含 4 个浮点数
    float y[4]={1,1,2,2};
    /* 向量 y 的元素为 1,1,2,2
     */

    // 调用 OpenBLAS 库的 cblas_saxpy 函数，将 alpha * x 加到 y 上
    // N 表示向量的长度，alpha 是标量，x 是源向量，1 是 x 的步长，y 是目标向量，1 是 y 的步长
    cblas_saxpy(N, alpha, x , 1, y, 1);

    // 创建一个 stringstream 对象，用于存储输出结果
    stringstream ss;
    // 保存当前的标准输出缓冲区
    streambuf   *buffer = cout.rdbuf();
    // 将标准输出重定向到 stringstream 对象
    cout.rdbuf(ss.rdbuf());
  
    // 遍历向量 y，将每个元素输出到 stringstream 对象
    for(int i=0;i<N;i++)
    {
        // 输出向量 y 的第 i 个元素，并在后面添加一个空格
        cout<<y[i]<<" ";
    }
  
    // 恢复标准输出缓冲区
    cout.rdbuf(buffer);
    // 将 stringstream 对象中的内容转换为字符串
    string s(ss.str());
    // 输出分隔线
    cout << "*********************************************************" << endl;
    // 输出测试结果的提示信息
    cout << "This is the result:" << endl;
    // 输出测试结果
    cout << s << endl;
    // 输出分隔线
    cout << "*********************************************************" << endl;
    // 输出参考结果的提示信息
    cout << "This is the reference:" << endl;
    // 输出参考结果
    cout << expect << endl;

    // 比较测试结果和参考结果
    if (expect == s)
        // 如果结果相同，输出测试通过的信息
        cout << "{Test PASS}." << endl;
    else
        // 如果结果不同，输出测试失败的信息
        cout << "{Test FAIL}." << endl;
    
    // 程序正常结束，返回 0
    return 0;
}

