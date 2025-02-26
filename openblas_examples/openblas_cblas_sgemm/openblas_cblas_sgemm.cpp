#include "cblas.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

string expect = R"(7 10 15 22 )";

#define M 2
#define N 2
#define K 2
/**
 * @brief 主函数，程序的入口点
 * 
 * 此函数使用 OpenBLAS 库中的 cblas_sgemm 函数进行单精度矩阵乘法操作，
 * 并将结果与预期结果进行比较，输出测试结果。
 * 
 * @param argc 命令行参数的数量
 * @param argv 命令行参数的数组
 * @return int 程序退出状态码，0 表示正常退出
 */
int main(int argc, char ** argv){
    // 定义矩阵乘法中的标量因子 alpha，这里设置为 1
    float alpha=1;
    // 定义矩阵乘法中的标量因子 beta，这里设置为 0
    float beta=0;
    // 定义矩阵 A 的 leading dimension，在按行优先存储时，lda 表示矩阵 A 的列数
    int lda=K;
    // 定义矩阵 B 的 leading dimension，在按行优先存储时，ldb 表示矩阵 B 的列数
    int ldb=N;
    // 定义矩阵 C 的 leading dimension，在按行优先存储时，ldc 表示矩阵 C 的列数
    int ldc=N;
    // 定义矩阵 A，使用一维数组存储二维矩阵，按行优先排列
    float A[M*K]={1,2,3,4};
    /* 矩阵 A 的实际形式为
     * 1,2
     * 3,4
     */
    // 定义矩阵 B，使用一维数组存储二维矩阵，按行优先排列
    float B[K*N]={1,2,3,4};
    /* 矩阵 B 的实际形式为
     * 1,2
     * 3,4
     */
    // 定义矩阵 C，用于存储矩阵乘法的结果
    float C[M*N];
    // 调用 OpenBLAS 库的 cblas_sgemm 函数进行矩阵乘法：C = alpha * A * B + beta * C
    // 参数依次为：矩阵存储顺序（行优先），矩阵 A 是否转置（不转置），矩阵 B 是否转置（不转置），
    // 矩阵 A 的行数，矩阵 B 的列数，矩阵 A 的列数（等于矩阵 B 的行数），标量因子 alpha，矩阵 A，
    // 矩阵 A 的 leading dimension，矩阵 B，矩阵 B 的 leading dimension，标量因子 beta，矩阵 C，
    // 矩阵 C 的 leading dimension
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    
    // 创建一个 stringstream 对象，用于存储矩阵 C 的输出结果
    stringstream ss;
    // 保存当前的标准输出缓冲区
    streambuf   *buffer = cout.rdbuf();
    // 将标准输出重定向到 stringstream 对象
    cout.rdbuf(ss.rdbuf());
    
    // 遍历矩阵 C 的每个元素
    for(int i=0;i<M*N;i++)
        // 输出矩阵 C 的第 i 个元素，并添加一个空格
        {cout<<C[i]<<" ";}
    
    // 恢复标准输出缓冲区
    cout.rdbuf(buffer);
    // 将 stringstream 对象中的内容转换为字符串
    string s(ss.str());
    // 输出分隔线
    cout << "*********************************************************" << endl;
    // 输出结果提示信息
    cout << "This is the result:" << endl;
    // 输出矩阵 C 的计算结果
    cout << s << endl;
    // 输出分隔线
    cout << "*********************************************************" << endl;
    // 输出参考结果提示信息
    cout << "This is the reference:" << endl;
    // 输出预期结果
    cout << expect << endl;

    // 比较实际结果和预期结果
    if (expect == s)
        // 如果结果相同，输出测试通过信息
        cout << "{Test PASS}." << endl;
    else
        // 如果结果不同，输出测试失败信息
        cout << "{Test FAIL}." << endl;

    // 程序正常结束，返回 0
    return 0;
}

