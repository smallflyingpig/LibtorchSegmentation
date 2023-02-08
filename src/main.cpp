#include<iostream>
#include"Segmentor.h"

int main(int argc, char *argv[])
{
	//auto model = UNet(1, "resnet34", "D:\\AllentFiles\\code\\tmp\\resnet34.pt");
	//model->to(at::kCUDA);
	//model->eval();
	//auto input = torch::rand({ 1,3,512,512 }).to(at::kCUDA);
	//auto output = model->forward(input);
	//int T = 100;
	//int64 t0 = cv::getCPUTickCount();
	//for (int i = 0; i < T; i++) {
	//	auto output = model->forward(input);
	//	//output = output.to(at::kCPU);
	//}
	//output = output.to(at::kCPU);
	//int64 t1 = cv::getCPUTickCount();
	//std::cout << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;
    std::vector<std::string> coco_name_list = {"BG", "person", "bicycle", "car", "motorcycle", "airplane",
                            "bus", "train", "truck", "boat", "traffic light",
                            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                            "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                            "kite", "baseball bat", "baseball glove", "skateboard",
                            "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                            "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                            "donut", "cake", "chair", "couch", "potted plant", "bed",
                            "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                            "keyboard", "cell phone", "microwave", "oven", "toaster",
                            "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"};
	std::vector<std::string> voc_name_list = {"__background__", "aeroplane", "bicycle",
							"bird","boat","bottle","bus","car","cat","chair","cow",
							"diningtable","dog","horse","motorbike","person","pottedplant",
							"sheep","sofa","train","tvmonitor"};
	std::vector<std::string> coco_class_list = {"person", "bicycle", "car", "motorcycle", "airplane", 
                                    "bus", "train", "truck", "boat", "bird", "cat", "dog", 
                                    "horse", "sheep", "cow", "elephant", "bear", "zebra", 
                                    "giraffe" };
	std::vector<std::string> voc_class_list = {"bicycle", "bird", "car", "cat", "dog", "person"};
	std::vector<std::string> person_class_list = {"person",};
    cv::Mat image = cv::imread("./voc_person_seg/val/2007_004000.jpg");

    Segmentor<DeepLabV3> segmentor;
    segmentor.Initialize(0,512,512, std::move(voc_name_list), //{"background","person"},
                         "resnet50","./weights/resnet50.pt");
    segmentor.LoadModule("./weights/deeplabv3_resnet50.pt");
    //segmentor.Predict(image,"person");
	segmentor.Predict(image, voc_class_list);

	//trainTricks tricks;

	////tricks for data augmentations
	//tricks.horizontal_flip_prob = 0.5;
	//tricks.vertical_flip_prob = 0.5;
	//tricks.scale_rotate_prob = 0.3;

	////tricks for training process
	//tricks.decay_epochs = { 40, 80 };
	//tricks.freeze_epochs = 8;

	//segmentor.SetTrainTricks(tricks);
    //segmentor.Train(0.0003,300,4,"D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg",".jpg","segmentor.pt");

    return 0;
}
