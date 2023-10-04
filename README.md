# Introduction
The widespread adoption of Federated Learning (FL), a privacy-preserving distributed learning methodology, has been impeded by the challenge of high communication overheads, typically arising from the transmission of large-scale models. Existing adaptive quantization methods, designed to mitigate these overheads, operate under the impractical assumption of uniform device participation in every training round. Additionally, these methods are limited in their adaptability due to the necessity of manual quantization level selection and often overlook biases inherent in local devices' data, thereby affecting the robustness of the global model. In response, this paper introduces AQUILA (adaptive quantization in device selection strategy), a novel adaptive framework devised to effectively handle these issues, enhancing the efficiency and robustness of FL. AQUILA integrates a sophisticated device selection method that prioritizes the quality and usefulness of device updates. Utilizing the exact global model stored by devices, it enables a more precise device selection criterion, reduces model deviation, and limits the need for hyperparameter adjustments. Furthermore, AQUILA presents an innovative quantization criterion, optimized to improve communication efficiency while assuring model convergence. Our experiments demonstrate that AQUILA significantly decreases communication costs compared to existing methods, while maintaining comparable model performance across diverse non-homogeneous FL settings, such as Non-IID data and heterogeneous model architectures.


## How to Run



## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@inproceedings{zhao2023inclusive,
  title={Inclusive Data Representation in Federated Learning: A Novel Approach Integrating Textual and Visual Prompt},
  author={Zhao, Zihao and Shi, Zhenpeng and Liu, Yang and Ding, Wenbo},
  booktitle={Adjunct Proceedings of the 2023 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings of the 2023 ACM International Symposium on Wearable Computers},
  year={2023}
}
```
