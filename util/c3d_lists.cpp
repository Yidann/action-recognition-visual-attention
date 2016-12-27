#include<iostream>
#include<algorithm>
#include<fstream>
#include<vector>
#include<string>
#include<cstring>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>  
#include <dirent.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


bool isDir(string dir) 
{
	struct stat fileInfo;
	stat(dir.c_str(), &fileInfo);

    if(S_ISDIR(fileInfo.st_mode)) 
	{
		return true;
	}

	else 
	{
		return false;
    }
}

void GetAllFormatFiles(string dir, vector<string> &files, bool recursive, string format) 
{
	DIR *dp; //create the directory object
	struct dirent *entry; //create the entry structure
	dp = opendir(dir.c_str());
	
	if(dir.at(dir.length()-1)!='/') 
	{
		dir=dir+"/";
	}
	if(dp!=NULL) 
	{
		while( entry=readdir(dp) ) 
		{
			string full_file_name = dir + entry->d_name;			
			if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ) 
			{
				if( isDir(full_file_name) == true && recursive == true) 
				{
					//files.push_back(string(entry->d_name)); //add entry to the list of files
					GetAllFormatFiles(full_file_name, files, true, format); //recurse
				} 
				else if( full_file_name.substr(full_file_name.find_last_of('.')) == format ) 
				{
					cout << "file name: " << full_file_name << endl;				
					files.push_back(full_file_name);//add the entry to the list of file
				} 
			}
		}
		closedir(dp); //close directory
	}
	else 
	{
		perror ("Couldn't open the directory.");
	}
}

int createPath( mode_t mode, const std::string& rootPath, std::string& path )
{
    struct stat st;

    for( string::iterator iter = path.begin()+rootPath.length() ; iter != path.end(); )
    {
         string::iterator newIter = find( iter, path.end(), '/' );
         string newPath = rootPath + "/" + string( path.begin(), newIter);

         if( stat( newPath.c_str(), &st) != 0)
         {           
             if( mkdir( newPath.c_str(), mode) != 0 && errno != EEXIST )
             {
                cout << "cannot create folder [" << newPath << "] : " << strerror(errno) << endl;
                return -1;
             }
         }
         else
            if( !S_ISDIR(st.st_mode) )
             {
                 errno = ENOTDIR;
                 cout << "path [" << newPath << "] not a dir " << endl;
                 return -1;
             }
             //else
             //    std::cout << "path [" << newPath << "] already exists " << endl;

         iter = newIter;
         if( newIter != path.end() )
             ++ iter;
    }
    return 0;
}

//replace old value to new value in a string
string& replace_all(string& str,const string& old_value,const string& new_value)   
{   
    if( old_value==new_value )
		return str;
	for(string::size_type pos(0); pos!=string::npos; pos+=new_value.length())   
	{   
        if( (pos=str.find(old_value,pos))!=string::npos )   
            str.replace(pos,old_value.length(),new_value);   
        else   
			break;   
    }   
    return   str;   
}


int main(int argc, char* argv[])
{
	assert(argc==2);
	string video(argv[1]);
	string video_dir = "/home/wyd/C3D/examples/c3d_feature_extraction/input/" + video;
	string root_dir = "/home/wyd/";
	size_t len = root_dir.length();
	int step_size = 8;

	vector<string> files;

	ofstream output_list;
	output_list.open("./lists/c3d/output_lists.txt", ios::trunc);
	ofstream input_list;
	input_list.open("./lists/c3d/input_lists.txt", ios::trunc);
	ofstream video_list;
	video_list.open("./lists/video_lists.txt",ios::trunc);

	//read avi video
	string format = ".avi";
	GetAllFormatFiles(video_dir, files, true, format);

	int size = files.size();
	cout << "size of video files: " << size << endl;
	double framenum = 0;
	string video_path;
	string output_folder;

	for(int i = 0;i<size;i++)
	{
		VideoCapture cap;
		cap.open(files[i]);
			if (!cap.isOpened())
			{
				cout << "can not open video:" <<files[i]<< endl;
				return -1;
			}
		framenum = cap.get(CV_CAP_PROP_FRAME_COUNT);

		if(framenum>16)
		{
			video_path = files[i];
			output_folder = video_path.substr( 0, (video_path.length()-format.length()) );
			replace_all(output_folder, "input", "output");
			//cout << "output folder: " << output_folder << endl;
			string output_path = output_folder.substr(len);
			//cout << "output path: " << output_path << endl;
			createPath(0777, "/home/wyd", output_path);

			char buffer[50]; 
			int chunknum=0;
			for(int start = 0; start<(framenum-16); start += step_size)
			{
				// <string path> <starting frame> <label>(0)				
				input_list<< video_path;
				input_list<< " ";
				input_list<< start;
				input_list<< " ";
				input_list<< "0" << endl;
				
				// <output_prefix>
				output_list<< output_folder;
				output_list<< "/";
				sprintf(buffer,"%06d", start);
				output_list<< buffer<<endl;
				chunknum++;
			}
			video_list << files[i]<<" "<< framenum << " " << chunknum << endl;
		}
		else
		{
			cout<<files[i]<<endl;
			cout<<"framenum is smaller than 16!"<<endl;
		}
	} 
	input_list.close();
	output_list.close();
	video_list.close();
	return 0;
}

