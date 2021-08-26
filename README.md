# 3D Lettuce Soil Segmentation 

## PointNet
### Examples from Test Set
<table>
    <thead>
        <tr>
            <th style="text-align: center;">Ground Truth</th>
            <th style="text-align: center;">Predicted</th>
            <th style="text-align: center;">Diff. (Error)</th>
        </tr>
    </thead>
    <tr>
        <td><img src='images/labeled_0.gif'></td>
        <td><img src='images/predicted_0.gif'></td>
        <td><img src='images/diff_0.gif'></td>
    </tr>
    <tr>
        <td><img src='images/labeled_1.gif'></td>
        <td><img src='images/predicted_1.gif'></td>
        <td><img src='images/diff_1.gif'></td>
    </tr>
    <tr>
        <td><img src='images/labeled_2.gif'></td>
        <td><img src='images/predicted_2.gif'></td>
        <td><img src='images/diff_2.gif'></td>
    </tr>
</table>

### Training Process
![Training Process](images/training.png)

### Accuracy
<table style="text-align: center;">
    <thead>
        <tr>
            <th>Train Set</th>
            <th>Validation Set</th>
            <th>Test Set</th>
        </tr>
    </thead>
    <tr>
        <td>97.55%</td>
        <td>95.63%</td>
        <td>95.228%</td>
    </tr>
</table>

<hr style='height: 10px; border: none; color: #333; background: linear-gradient(90deg, #232526 0%, #414345 100%);'>

## RandLA-Net
### Examples from Test Set
<table>
    <thead>
        <tr>
            <th style="text-align: center;">Ground Truth</th>
            <th style="text-align: center;">Predicted</th>
            <th style="text-align: center;">Diff. (Error)</th>
        </tr>
    </thead>
    <tr>
        <td><img src='images/RandLANet_labeled_0.gif'></td>
        <td><img src='images/RandLANet_predicted_0.gif'></td>
        <td><img src='images/RandLANet_diff_0.gif'></td>
    </tr>
    <tr>
        <td><img src='images/RandLANet_labeled_1.gif'></td>
        <td><img src='images/RandLANet_predicted_1.gif'></td>
        <td><img src='images/RandLANet_diff_1.gif'></td>
    </tr>
    <tr>
        <td><img src='images/RandLANet_labeled_2.gif'></td>
        <td><img src='images/RandLANet_predicted_2.gif'></td>
        <td><img src='images/RandLANet_diff_2.gif'></td>
    </tr>
</table>

### Training Process
![Training Process](images/training_RandLANet.png)

### Accuracy
<table style="text-align: center;">
    <thead>
        <tr>
            <th>Train Set</th>
            <th>Validation Set</th>
            <th>Test Set</th>
        </tr>
    </thead>
    <tr>
        <td>97.35%</td>
        <td>96.48%</td>
        <td>97.2%</td>
    </tr>
</table>

<hr style='height: 10px; border: none; color: #333; background: linear-gradient(90deg, #232526 0%, #414345 100%);'>

## PointNet++
### Examples from Test Set
<table>
    <thead>
        <tr>
            <th style="text-align: center;">Ground Truth</th>
            <th style="text-align: center;">Predicted</th>
            <th style="text-align: center;">Diff. (Error)</th>
        </tr>
    </thead>
    <tr>
        <td><img src='images/PointNet2_labeled_0.gif'></td>
        <td><img src='images/PointNet2_predicted_0.gif'></td>
        <td><img src='images/PointNet2_diff_0.gif'></td>
    </tr>
    <tr>
        <td><img src='images/PointNet2_labeled_1.gif'></td>
        <td><img src='images/PointNet2_predicted_1.gif'></td>
        <td><img src='images/PointNet2_diff_1.gif'></td>
    </tr>
    <tr>
        <td><img src='images/PointNet2_labeled_2.gif'></td>
        <td><img src='images/PointNet2_predicted_2.gif'></td>
        <td><img src='images/PointNet2_diff_2.gif'></td>
    </tr>
</table>

### Training Process
![Training Process](images/training_PointNet2.png)

### Accuracy
<table style="text-align: center;">
    <thead>
        <tr>
            <th>Train Set</th>
            <th>Validation Set</th>
            <th>Test Set</th>
        </tr>
    </thead>
    <tr>
        <td>98.40%</td>
        <td>97.55%</td>
        <td>98.32%</td>
    </tr>
</table>
