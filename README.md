# unet-orthoimagery
## Members
  - Philip Englund Mathieu
  - GunGyeom James Kim
  - Hao Sheng Ning
## Project Description
  - This project demonstrates a method for segmenting aerial imagery using the U-Net CNN to detect the presence of Fallopia japonica (Japanese knotweed), a plant species that is categorized as invasive in Maine. The project is based on a paper that achieved a similar goal using unmanned aerial vehicle (UAV, i.e. drone) imagery to detect different plant species in a different region of the US. This project utilizes ortho-rectified aerial imagery obtained from the Maine GeoLibrary which is freely available. If successful, the proposed solution will enable early detection, mapping, and monitoring of invasive plants in Maine, thereby improving the state's ecological, social, and economic conditions.
## URL for Presentation
  - 
## URL(s) for Others
  - None

## etc.
### Requirements
- torch
- cv2
- matplotlib: 3.6.2
- numpy: 1.23.5
- Pillow: 9.3.0
- tqdm: 4.64.1
- wandb: 0.13.5

### References
- [milesial/Pytorch-UNet, Github](https://github.com/milesial/Pytorch-UNet)
- [Sorensen-Dice coefficient, Wikipedia](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [RMSProp, Cornell Wiki](https://optimization.cbe.cornell.edu/index.php?title=RMSProp#:~:text=RMSProp%2C%20root%20mean%20square%20propagation,lecture%20six%20by%20Geoff%20Hinton.)
- [Explanation of why Dice coefficient works for imbalanced data](https://stats.stackexchange.com/questions/438494/what-is-the-intuition-behind-what-makes-dice-coefficient-handle-imbalanced-data)
