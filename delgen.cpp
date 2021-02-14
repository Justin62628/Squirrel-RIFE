#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <Windows.h>
#include <ctime>
#include <queue>
#include <thread>

using namespace cv;
using namespace std;

string path;
double dup;
double scene;
int total;
int thread_tasks[2][129];
bool done[129];
queue<int> q[129];
int T;
int ttttt=0;

string topath(int i)
{
	string s = "";
	while (i != 0)
	{
		s = char(i % 10 + '0') + s;
		i /= 10;
	}
	while (s.length() < 9)
		s = '0' + s;
	string path1;
	path1 = path;
	path1 = path1 + '\\';
	path1 = path1 + s;
	path1 = path1 + ".png";
	return path1;
}
double ave(Mat a)
{
	double sum = 0;
	int n = 0;
	for (int i = 0; i <= 3; i++)
	{
		double tem = mean(a).val[i];
		sum += tem;
		if (tem != 0.0)
			n++;
	}
	if (n == 0)
		return 0.0;
	return sum / n;
}
double diff(int img1, int img2)
{
	Mat i1 = imread(topath(img1));
	Mat i2 = imread(topath(img2));
	Mat i3;
	if (i1.empty() || i2.empty())
		return 0.0;
	absdiff(i1, i2, i3);
	return ave(i3);
}
void thread_task(int start,int end,int number)
{
	int st = start - 5;
	if (st < 1)
		st = 1;
	int pos = st;
	while (pos <= end)
	{
		pos++;
		ttttt++;
		double dif = diff(st, pos);
		while (dif < dup&&pos <= end)
		{
			dif = diff(st, pos + 1);
			if (dif < scene&&pos>=start)
				q[number].push(pos);
			pos++;
			ttttt++;
		}
		st = pos;
	}
	done[number] = true;
	return;
}
bool getdone()
{
	bool a = true;
	for (int i = 1; i <= T; i++)
		a &= done[i];
	return a;
}
int main()
{
	getline(cin, path);
	if (path[0] == '"')
	{
		path.erase(0, 1);
		path.erase(path.end() - 1);
	}
	cin >> dup >> scene >> total >>T;
	total--;
	int tempp = total / T;
	for (int i = 1; i <= T; i++)
	{
		thread_tasks[0][i] = thread_tasks[1][i - 1];
		thread_tasks[1][i] = thread_tasks[0][i] + tempp;
	}
	thread_tasks[1][T] = total;
	thread_tasks[0][1] = 1;
	thread t[129];
	for (int i = 1; i <= T; i++)
	{
		t[i - 1] = thread(thread_task, thread_tasks[0][i], thread_tasks[1][i],i);
		t[i - 1].detach();
	}
	int ttt = ttttt;
	while (!getdone())
	{
		cout << ttttt << "/" << total+5*T+1 <<",speed="<<(ttttt-ttt)*5<<'x'<< endl;
		ttt = ttttt;
		Sleep(200);
	}
	for (int i = 1; i <= T; i++)
	{
		while (q[i].size())
		{
			string ss = topath(q[i].front());
			q[i].pop();
			char s[1000];
			for (int w = 0; w < ss.length(); w++)
				s[w] = ss[w];
			s[ss.length()] = '\0';
			remove(s);
		}
	}
	cout << "Done!";
	return 0;
}
