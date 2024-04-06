from DeepImageSearch import Load_Data, Search_Setup

dl = Load_Data()

image_list = dl.from_folder(["images2"])
image_list[:5]

st = Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None)

st.run_index


