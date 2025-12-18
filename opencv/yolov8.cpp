#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<iostream>
#include<vector>
#include<string>

using std::vector;
using cv::Mat;
using std::endl;
using std::cout;
using std::string;


void print_result(const Mat&result,float conf=0.7,int len_data=15)//每一行多少数据，默认15
{
    float *pdata =(float*)result.data;

    for(int i=0;i<result.total()/len_data;i++)
    {
        if(pdata[4]>conf)
        {
            for(int j=0;j<len_data;j++)
            {
                cout<<pdata[j]<<" ";
            }
            cout<<endl;
        }
        pdata+=len_data;
    }
    return ;

}
//筛除执行度过低的目标
vector<vector<float>>get_info(const Mat&result,float conf=0.7,int len_data=15)//每一行多少数据，默认15
{
    float *pdata =(float*)result.data;
    vector<vector<float>>info;
    for(int i=0;i<result.total()/len_data;i++)
    {
        if(pdata[4]>conf)
        {
            vector<float>info_line;
            for(int j=0;j<len_data;j++)
            {
                info_line.push_back(pdata[j]);
            }
            info.push_back(info_line);
        }
        pdata+=len_data;
    }
    return info;
}

vector<vector<vector<float>>>group_info(const vector<vector<float>>&info,int num_classes=80)
{
    vector<vector<vector<float>>>grouped_info;
    grouped_info.resize(num_classes);
    for(auto i=0;i<info.size();i++)
    {
        int class_id=static_cast<int>(info[i][5]);
        grouped_info[class_id].push_back(info[i]);
    }
    return grouped_info;
}

void info_simplify(vector<vector<float>>&info)
{
    for(auto i=0;i<info.size();i++)
    {
        info[i][5]=*std::max_element(info[i].cbegin()+5,info[i].cend());
        info[i].resize(6);
        float x=info[i][0];
        float y=info[i][1];
        float w=info[i][2];
        float h=info[i][3];
        info[i][0]=x-(w/2);
        info[i][1]=y-(h/2);
        info[i][2]=x+(w/2);
        info[i][3]=y+(h/2);
    }
}

void print_info(const vector<vector<float>>&info)
{
    for(auto i=0;i<info.size();i++)
    {
        for(auto j=0;j<info[i].size();j++)
        {
            cout<<info[i][j]<<endl;
        }
        cout<<endl;
    }
}
int main()
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("onnx模型");
    cv::Mat img=cv::imread("图片路径");
    cv::resize(img,img,cv::Size(640,640));
    cv::Mat blob=cv::dnn::blobFromImage(img,1.0/255.0,cv::Size(640,640),cv::Scalar(),true);
    net.setInput(blob);
    vector<cv::Mat>netoutput;
    vector<string>out_name={"output"};
    net.forward(netoutput,out_name);
    Mat result=netoutput[0];
    //print_result(result);
    vector<vector<float>>info=get_info(result);
    info_simplify(info);
    print_info(info);
    cout<<info.size()<<" "<<info[0].size()<<endl;
    return 0;
}