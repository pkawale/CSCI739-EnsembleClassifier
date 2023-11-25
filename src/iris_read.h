#include<iostream>
#include<algorithm>
#include<sstream>
#include<fstream>
#include<cstring>

enum flower {Iris_setosa, Iris_versicolor, Iris_virginica, Iris_unkown};

void Read_Iris_Dataset(std::vector<std::vector<double>>& dataset, 
                        std::vector<int>& labels)
{
   std::ifstream myfile("iris.data");
  std::string line;
  float sepal_len_f,sepal_wid_f,petal_len_f,petal_wid_f;

  float iris_class_f;

  std::string temp_string;
   int count =0;
   if (myfile.is_open())
  {
     std::cout<< "file opened successfully"<<std::endl;
      while (std::getline(myfile, line)) {
         std::replace(line.begin(), line.end(), '-', '_');
         std::replace(line.begin(), line.end(), ',', ' ');
         
         std::istringstream iss(line);
         count++;
         
         std::vector<double> temp_row;
         iss >> sepal_len_f>>sepal_wid_f >> petal_len_f >>petal_wid_f >> temp_string;
         temp_row.push_back(sepal_len_f);
         temp_row.push_back(sepal_wid_f);
         temp_row.push_back(petal_len_f);
         temp_row.push_back(petal_wid_f);
         dataset.push_back(temp_row);

         if(temp_string.compare("Iris_setosa") == 0)
         {
            iris_class_f = Iris_setosa;
         }
         else if (temp_string.compare("Iris_versicolor") == 0)
         {
            iris_class_f = Iris_versicolor;
         }
         else if (temp_string.compare("Iris_virginica") == 0)
         {
            iris_class_f = Iris_virginica;
         }else 
         {
            iris_class_f = Iris_unkown;
         }
         labels.push_back(int(iris_class_f));
      }  
  }
  else 
  {
     std::cout << "Unable to open file";
  }

}