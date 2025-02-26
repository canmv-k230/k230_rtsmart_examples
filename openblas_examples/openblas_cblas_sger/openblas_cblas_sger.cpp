#include "cblas.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

string expect = R"(20 40 10 20 30 60 )";

/**
 * @brief 主函数，程序入口
 * 
 * 此函数用于测试 OpenBLAS 库中的 cblas_sger 函数。
 * 该函数执行矩阵更新操作 A <== alpha * x * y' + A，其中 y' 是 y 的转置。
 * 程序会将结果与预期结果进行比较，并输出测试结果。
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序退出状态码，0 表示正常退出
 */
int main(int argc, char ** argv)
{
    // 定义向量 x，包含 2 个浮点数
    float x[2] = {1.0, 2.0};
    // 定义向量 y，包含 3 个浮点数
    float y[3] = {2.0, 1.0, 3.0};
    // 定义矩阵 A，初始化为全 0，矩阵大小为 2x3
    float A[6] = { 0 };
    // 定义矩阵 A 的行数
    blasint rows = 2, 
    // 定义矩阵 A 的列数
            cols = 3;
    // 定义标量 alpha，用于向量乘法的缩放因子
    float alpha = 10;
    // 定义向量 x 的增量，用于指定元素访问的步长
    blasint inc_x = 1, 
    // 定义向量 y 的增量，用于指定元素访问的步长
            inc_y = 1;
    // 定义矩阵 A 的 leading dimension，在列优先存储中，lda 表示列的长度
    blasint lda = 2;

    // 矩阵按列优先存储
    // 执行矩阵更新操作 A <== alpha * x * y' + A （y' 表示 y 的转置）
    cblas_sger(CblasColMajor, rows, cols, alpha, x, inc_x, y, inc_y, A, lda);
    
    // 创建一个 stringstream 对象，用于存储矩阵 A 的输出结果
    stringstream ss;
    // 保存当前的标准输出缓冲区
    streambuf   *buffer = cout.rdbuf();
    // 将标准输出重定向到 stringstream 对象
    cout.rdbuf(ss.rdbuf());
  
    // 遍历矩阵 A 的每一行
    for(int i=0;i<rows;i++)
    {
        // 遍历矩阵 A 的每一列
        for(int j=0;j<cols;j++)
        {
            // 输出矩阵 A 中第 i 行第 j 列的元素，并添加一个空格
            cout<<A[i*cols+j]<<" ";
        }
    }
  
    // 恢复标准输出缓冲区
    cout.rdbuf(buffer);
    // 将 stringstream 对象中的内容转换为字符串
    string s(ss.str());
    // 输出分隔线
    cout << "*********************************************************" << endl;
    // 输出结果提示信息
    cout << "This is the result:" << endl;
    // 输出矩阵 A 的结果
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

