# Weighted (Rank) & Saturation Time (Metric) with Adaptive (Schedule) Runs

for model in resnet-18 resnet-34 resnet-50 resnet-101 resnet-152; do

    for dataset in mnist cifar-10 cifar-100; do

        gradus train $model $dataset --rank weighted --metric saturation-time --schedule adaptive

    done

done