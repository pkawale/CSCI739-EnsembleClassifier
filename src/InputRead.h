#include <vector>
#include <iostream>
#include <fstream>

typedef unsigned char uchar;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void read_mnist_params(std::string mnist_parameter_file, std::vector<std::vector<double>> &arr)
{
    std::ifstream file (mnist_parameter_file,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        std::cout<<mnist_parameter_file<<"images: "<<number_of_images<< "\tn_rows: "<<n_rows<<"\tn_cols: "<<n_cols<<std::endl;

        arr.resize(number_of_images,std::vector<double>(n_rows * n_cols));
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}


void read_mnist_labels(std::string full_path, std::vector<std::vector<double>>& dataset) {

    std::ifstream file(full_path, std::ios::binary);
    // std::vector<double> labels;
    int number_of_labels;

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        // if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = ReverseInt(number_of_labels);

        std::cout<<"Labels: "<< number_of_labels<<std::endl;

        unsigned char temp=0;
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&temp, 1);
            dataset[i].push_back(double(temp));
        }
        return;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void read_mnist(std::string feature_path, std::string label_path, std::vector<std::vector<double>>& dataset){
    read_mnist_params(feature_path, dataset);
    read_mnist_labels(label_path, dataset);

}

// int main(){

//     std::vector<std::vector<double>> dataset;
//     // read_mnist_params("train-images-idx3-ubyte", dataset);
//     read_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", dataset);
//     for(size_t i=0; i<3; ++i){
//         for(auto x:dataset[i])
//             std::cout<<x<< " ";
//     }
//     std::cout<<std::endl;
// }