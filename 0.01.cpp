//I killed it.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <Windows.h>
#include <ctime>
#include <thread>

using namespace cv;
using namespace std;
bool tag = false;
class Queue
{
	public:
		int pp[1010];
		Mat q[510];
		int hh;
		int tt;
		int end;
		bool ee;
		void init(void);
		void push_back(Mat a);
		Mat get_front(void);
		bool empty(void);
		void toll(string path);
		bool full(void);
		int si(void);
};
int Queue::si(void)
{
	if (hh <= tt)
	{
		return tt - hh;
	}
	else
		return 500 + tt - hh;
}
bool Queue::full(void)
{
	if ((tt == 499) && (hh == 0)) return true;
	if (tt == hh - 1) return true;
	return false;
}
void Queue::init(void)
{
	hh = tt = 0;
	end = 0;
	ee = false;
}
void Queue::push_back(Mat a)
{
	q[tt++] = a;
	if (tt == 500)
		tt = 0;
}
Mat Queue::get_front(void)
{
	Mat a=q[hh++];
	if (hh == 500)
	{
		hh = 0;
	}
	return a;
}
bool Queue::empty(void)
{
	return hh == tt;
}
string tostring(int i)
{
	string s = "";
	while (i != 0)
	{
		s = char(i % 10 + '0') + s;
		i /= 10;
	}
	while (s.length() < 10)
	{
		s = '0' + s;
	}
	return s;
}
double ave(Mat a)
{
	double sum = 0;
	int n=0;
	for (int i = 0; i <= 3; i++)
	{
		double tem = mean(a).val[i];
		sum += tem;
		if (tem != 0.0)
		{
			n++;
		}
	}
	return sum / n;
}
Queue qq;
double diff(int img1, int img2)
{
	Mat a;
	try 
	{
		absdiff(qq.q[img1], qq.q[img2], a);
		return ave(a);
	}
	catch(...)
	{
		cout << time(0);
		tag = true;
		return 0.0;
	}
}
void Queue::toll(string path)
{
	string path1=path;
	if (ee)
	{
		return;
	}
	while(!full())
	{
		end++;
		string s = tostring(end);
		path = path1;
		path = path + "\\";
		path = path + s;
		path = path + ".png";
		Mat a = imread(path);
		Mat b;
		push_back(a);
		int bt;
		if (tt != 0)
			bt = tt - 1;
		else
		{
			bt = 499;
		}
		pp[bt] = end;
		if (q[bt].empty())
		{
			ee = true;
			return;
		}
	}
}
string imgs;
void threadtask()
{
	while (!qq.ee)
	{
		if (qq.si() < 400)
		{
			qq.toll(imgs);
		}
	}
	return;
}
int main()
{
	FILE* stream1;
	qq.init();
	double dup;
	double scene;
	cin >> imgs >> dup >> scene;
	qq.toll(imgs);
	thread t(threadtask);
	cout << "1";
	freopen_s(&stream1, "D:\\log.txt", "w", stdout);
	cout << time(0) << endl;
	while (!qq.empty() || !qq.ee)
	{
		
		if (qq.empty())
		{
			Sleep(500);
		}
		int pos = qq.hh;
		pos++;
		if (pos == 500) pos = 0;
		if (pos == qq.tt)
		{
			Sleep(500);
		}
		if (qq.empty() && qq.ee)
		{
			return 0;
		}
		double ddd = diff(qq.hh, pos);
		if (tag)
		{
			return 0;
		}
		while (ddd < dup)
		{
			if (pos == 500) pos = 0;
			if (pos == qq.tt)
			{
				Sleep(500);
			}
			if (qq.empty() && qq.ee)
			{
				return 0;
			}
			ddd = diff(qq.hh, pos==499?0:pos+1);
			if (tag)
			{
				return 0;
			}
			if (ddd < scene)
			{
				cout<<tostring(int(qq.pp[pos]))<<endl;
			}
			pos++;
			if (pos == 500) pos = 0;
			if (pos == qq.tt)
			{
				Sleep(100);
			}
			if (qq.empty() && qq.ee)
			{
				return 0;
			}
		}
		qq.hh = pos;
		if (pos == qq.tt)
		{
			Sleep(500);
		}
	}
	cout << time(0) << endl;
	fclose(stdout);
	return 0;
}
