# Weighted (Rank) & Saturation Time (Metric) Runs

for model in resnet-18 resnet-34 resnet-50 resnet-101 resnet-152; do

    for dataset in mnist cifar-10 cifar-100; do

        gradus train $model $dataset --rank weighted --metric saturation-time

    done

done