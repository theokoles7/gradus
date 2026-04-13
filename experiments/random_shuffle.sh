# Random Shuffle Runs

for model in resnet-18 resnet-34 resnet-50 resnet-101 resnet-152 vgg-11 vgg-13 vgg-16 vgg-19; do

    for dataset in mnist cifar-10 cifar-100; do

        gradus train $model $dataset --shuffle

    done

done