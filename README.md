data from here https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
python train.py
If the data is in a different directory, you can specify the path at runtime by using the --data_dir flag:

python train.py --data_dir <path/to/data>
To see the parameters that can be changed, run

python train.py -h
The training loss will be updated in the logs folder which can be run with tensorboard to visualise the generator and discriminator loss on the browser. Run this command:

tensorboard --logdir=logs
At the end of training, the output of this code is the model weights of the two generators and the two discriminators. These will be saved as:

generatorAToB.h5 - 43.5MB
generatorBToA.h5 - 43.5MB
discriminatorA.h5 - 10.5MB
discriminatorB.h5 - 10.5MB
Testing
Once the model is trained, the model file should be in the same folder as the test script with the name: generatorAToB.h5 and generatorBToA.h5. The test data shold be in --data_dir parameter in the folder structure as described above. Using the --batch_size flag, you can specify home many test images should get modified.

You can test the model by running:

python test.py
The results will be generated in results folder. Create two folders m2f and f2m inside the results folder. The corresponding transformations will appear in these two folders. The results will contain - fake, real, reconstructed and identity images.
