#include "stdio.h"
#include "stdlib.h"
#include "f77blas.h"
#include "sys/time.h"
#include "time.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

extern void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);

string expect = R"(16.801 18.002 18.003 16.801 15.602 22.803 )";

/**
 * @brief 主函数，程序入口点
 * 
 * 此函数用于执行矩阵乘法操作，使用 Fortran 风格的 BLAS 函数 dgemm_ 进行双精度矩阵乘法。
 * 它会初始化矩阵 A、B 和 C，调用 dgemm_ 函数进行计算，然后输出结果并与预期结果进行比较。
 * 
 * @param argc 命令行参数的数量
 * @param argv 命令行参数的数组
 * @return int 程序退出状态码，0 表示正常退出
 */
int main(int argc, char* argv[])
{
  // 循环计数器
  int i = 0;
  // 矩阵 A 的行数，矩阵 C 的行数
  int m = 2;
  // 矩阵 B 的列数，矩阵 C 的列数
  int n = 3;
  // 矩阵 A 的列数，矩阵 B 的行数
  int k = 4;
  // 矩阵 A 的元素个数
  int sizeofa = m * k;
  // 矩阵 B 的元素个数
  int sizeofb = k * n;
  // 矩阵 C 的元素个数
  int sizeofc = m * n;
  // 矩阵 A 是否转置的标志，'N' 表示不转置
  char ta = 'N';
  // 矩阵 B 是否转置的标志，'N' 表示不转置
  char tb = 'N';
  // 矩阵乘法中的标量因子 alpha
  double alpha = 1.2;
  // 矩阵乘法中的标量因子 beta
  double beta = 0.001;

  // 动态分配内存给矩阵 A
  double* A = (double*)malloc(sizeof(double) * sizeofa);
  // 动态分配内存给矩阵 B
  double* B = (double*)malloc(sizeof(double) * sizeofb);
  // 动态分配内存给矩阵 C
  double* C = (double*)malloc(sizeof(double) * sizeofc);

  // 初始化随机数种子
  srand((unsigned)time(NULL));

  // 初始化矩阵 A 的元素
  for (i=0; i<sizeofa; i++)
    // 矩阵 A 的元素初始化为 i % 3 + 1
    A[i] = i%3+1;//(rand()%100)/10.0;

  // 初始化矩阵 B 的元素
  for (i=0; i<sizeofb; i++)
    // 矩阵 B 的元素初始化为 i % 3 + 1
    B[i] = i%3+1;//(rand()%100)/10.0;

  // 初始化矩阵 C 的元素
  for (i=0; i<sizeofc; i++)
    // 矩阵 C 的元素初始化为 i % 3 + 1
    C[i] = i%3+1;//(rand()%100)/10.0;

  // 输出矩阵乘法的参数信息
  printf("m=%d,n=%d,k=%d,alpha=%lf,beta=%lf,sizeofc=%d\n",m,n,k,alpha,beta,sizeofc);
  // 调用 dgemm_ 函数进行矩阵乘法：C = alpha * A * B + beta * C
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);

  // 输出矩阵 A 的信息
  printf("This is matrix A\n\n");
  for(i=0; i < sizeofa; i++)
    // 输出矩阵 A 的每个元素
    printf("%lf ", A[i]);
  printf("\n");
  
  // 输出矩阵 B 的信息
  printf("This is matrix B\n\n");
  for(i=0; i < sizeofb; i++)
    // 输出矩阵 B 的每个元素
    printf("%lf ", B[i]);
  printf("\n");
  
  // 创建一个 stringstream 对象，用于存储矩阵 C 的输出结果
  stringstream ss;
  // 保存当前的标准输出缓冲区
  streambuf   *buffer = cout.rdbuf();
  // 将标准输出重定向到 stringstream 对象
  cout.rdbuf(ss.rdbuf());
  
  // 遍历矩阵 C 的每个元素
  for(int i=0;i<sizeofc;i++)
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

  // 释放矩阵 A 的内存
  free(A);
  // 释放矩阵 B 的内存
  free(B);
  // 释放矩阵 C 的内存
  free(C);
  // 程序正常结束，返回 0
  return 0;
}
