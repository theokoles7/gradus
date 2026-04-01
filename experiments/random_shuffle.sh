# Random Shuffle Runs

for model in resnet-18 resnet-34 resnet-50 resnet-101 resnet-152 vgg-16 vgg-19; do

    for dataset in mnist cifar-10 cifar-100; do

        gradus train --epochs 200 $model $dataset --batch-size 128 --shuffle

    done

done