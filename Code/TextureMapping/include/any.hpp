#pragma once

#ifndef ANY_HPP
#define ANY_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <memory>
#include <string>

#include <helper_math.h>
#include <config_utils.h>

using namespace std;


static string currentTime() {
	string total_time = "_";
	time_t now = time(0);
	tm *ltm = localtime(&now);
	ostringstream ossM, ossD, ossH, ossm;
	ossm << setw(2) << setfill('0') << ltm->tm_min;
	ossH << setw(2) << setfill('0') << ltm->tm_hour;
	ossD << setw(2) << setfill('0') << ltm->tm_mday;
	ossM << setw(2) << setfill('0') << ltm->tm_mon + 1;
	total_time += ossM.str() + ossD.str() + ossH.str() + ossm.str();
	return total_time;
}

static void printProgBar(int percent) {
	string bar;

	for (int i = 0; i < 50; i++) {
		if (i < (percent / 2)) {
			bar.replace(i, 1, "=");
		}
		else if (i == (percent / 2)) {
			bar.replace(i, 1, ">");
		}
		else {
			bar.replace(i, 1, " ");
		}
	}

	cout << "\r" "[" << bar << "] ";
	cout.width(3);
	cout << percent << "%     " << std::flush;
}

static string zeroPadding(string str, const size_t num) {
	if (num > str.size())
		str.insert(0, num - str.size(), '0');
	return str;
}
static string zeroPadding(int tar_num, const size_t num) {
	string str = to_string(tar_num);
	if (num > str.size())
		str.insert(0, num - str.size(), '0');
	return str;
}

static void shirink_grow(float2* points, float ratio) {
	float2 center = (points[0] + points[1] + points[2]) / 3.0;
	for (int i = 0; i < 3; i++) {
		points[i] += (center - points[i])*ratio;
	}
}


#endif