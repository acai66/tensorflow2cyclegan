import networks
import helpers
from helpers import cfg
import time
import os.path
import tensorflow as tf
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchSize', type=int, default=2, help='Batch Size to be used for training')
    parser.add_argument('--testBatchSize', type=int, default=10, help='Batch Size to be used for testing')
    parser.add_argument('--max_steps', type=int, default=1000000, help='Number of max_steps that training should run')
    parser.add_argument('--lambda_cyc', type=float, default=10, help='lambda value for cycle consistency loss')
    parser.add_argument('--lambda_idt', type=float, default=5, help='lambda value for identity loss')
    parser.add_argument('--print_steps_freq', type=int, default=20, help='The frequency of printing loss and acc')
    parser.add_argument('--tensorboard_images_freq', type=int, default=0, help='The frequency of Tensorboard images updating, 0 means do not write images to logs')
    parser.add_argument('--save_steps_freq', type=int, default=10000, help='The frequency at which model should be saved and evaluated')
    parser.add_argument('--num_resnet_blocks', type=int, default=3, help='Number of ResNet blocks for transformation in generator')
    parser.add_argument('--data_dir', type=str, default='data/vangogh2photo', help='Directory where train and test images are present')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for adversarial model')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='Learning rate for discriminators')
    parser.add_argument('--resume', type=int, default=0, help='Resume from saved weights')

    opt, _ = parser.parse_known_args()

    data_dir = opt.data_dir #"data/captcha/"
    batch_size = opt.batchSize # 32
    test_batch_size = opt.testBatchSize # 32
    max_steps = opt.max_steps #40
    lambda_cyc = opt.lambda_cyc #10
    lambda_idt = opt.lambda_idt #5
    print_steps_freq = opt.print_steps_freq # 20
    tensorboard_images_freq = opt.tensorboard_images_freq # 0
    save_steps_freq = opt.save_steps_freq #10000
    num_resnet_blocks = opt.num_resnet_blocks #5
    lr = opt.lr
    d_lr = opt.d_lr
    resume = opt.resume

    trainA, trainB = helpers.load_train_images(data_dir, batch_size)
    testA, testB = helpers.load_test_images(data_dir, test_batch_size)

    #Define the two discriminator models
    discA = networks.define_discriminator_network()
    discB = networks.define_discriminator_network()
    discA_optimizer = Adam( d_lr, 0.5)
    discB_optimizer = Adam( d_lr, 0.5)

    print(discA.summary())

    # The discriminators are trained on MSE loss on the batch output
    # Compile the model for dicriminators
    discA.compile(optimizer=discA_optimizer, loss='mse', metrics= ['accuracy'])
    discB.compile(optimizer=discB_optimizer, loss='mse', metrics= ['accuracy'])

    real_labels = tf.ones_like((batch_size, ))
    fake_labels = tf.zeros_like((batch_size, ))

    # Define the two generator models
    genA2B = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)
    genB2A = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)

    print(genA2B.summary())    

    # make the dicriminators non-trainable in the adversarial model
    discA.trainable = False
    discB.trainable = False

    # Define the adversarial model
    train_optimizer = Adam( lr, 0.5)
    gan_model = networks.define_adversarial_model(genA2B, genB2A, discA, discB, train_optimizer, lambda_cyc=lambda_cyc, lambda_idt=lambda_idt)

    # Setup the tensorboard to store and visualise losses
    if not os.path.exists('logs'):
        os.mkdir('logs')
    tensorboard = TensorBoard(log_dir=os.path.join('logs', '{}'.format(time.time())), write_images=True, 
                                  write_graph=True)
    tensorboard.set_model(gan_model)
    print("Batch Size: {}".format(batch_size))
    print("Num of ResNet Blocks: {}".format(num_resnet_blocks))
    print("Starting training for {0} steps with lambda_cyc = {1}, lambda_idt = {2}, num_resnet_blocks = {3}".format(max_steps, lambda_cyc, lambda_idt, num_resnet_blocks))

    if resume != 0:
        genA2B.load_weights(os.path.join('weight', 'generatorAToB_temp_%d.h5' % resume))
        genB2A.load_weights(os.path.join('weight', 'generatorAToB_temp_%d.h5' % resume))
        discA.load_weights(os.path.join('weight', 'discriminatorA_temp_%d.h5' % resume))
        discB.load_weights(os.path.join('weight', 'discriminatorB_temp_%d.h5' % resume))
    
    # Start training
    writer = tf.summary.create_file_writer('logs')
    with writer.as_default():
        for step in range(resume, max_steps):

            # Sample images
            realA = tf.image.resize(next(trainA), size=(cfg['height'], cfg['width']))
            realB = tf.image.resize(next(trainB), size=(cfg['height'], cfg['width']))

            # Translate images to opposite domain
            fakeB = genA2B(realA)
            fakeA = genB2A(realB)

            # Train the discriminator A on real and fake images
            dLossA_real = discA.train_on_batch(realA, real_labels)
            dLossA_fake = discA.train_on_batch(fakeA, fake_labels)

            # Train the discriminator B on ral and fake images
            dLossB_real = discB.train_on_batch(realB, real_labels)
            dLossB_fake = discB.train_on_batch(fakeB, fake_labels)


            # Train the generator networks
            g_loss = gan_model.train_on_batch([realA, realB], [real_labels, real_labels, realA, realB, realA, realB])


            
            if step % print_steps_freq == 0:
                # Calculate the total discriminator loss
                mean_discA_loss, mean_discA_acc = 0.5 * tf.add(dLossA_real, dLossA_fake)
                mean_discB_loss, mean_discB_acc = 0.5 * tf.add(dLossB_real, dLossB_fake)
                print("step: {}\tDiscA_Loss:{}\tDiscA_Acc:{}\tDiscB_Loss:{}\tDiscB_Acc:{}\tAdversarial Model losses:{}".format(step, mean_discA_loss, mean_discA_acc, mean_discB_loss, mean_discB_acc, g_loss[0]))
                tf.summary.scalar("mean_discA_Loss", mean_discA_loss, step=step)
                tf.summary.scalar("mean_discA_Acc", mean_discA_acc, step=step)
                tf.summary.scalar("mean_discB_Loss", mean_discB_loss, step=step)
                tf.summary.scalar("mean_discB_Acc", mean_discB_acc, step=step)
                tf.summary.scalar("decisionA_Loss", g_loss[1], step=step)
                tf.summary.scalar("decisionB_Loss", g_loss[2], step=step)
                tf.summary.scalar("reconstructedA_Los", g_loss[3], step=step)
                tf.summary.scalar("reconstructedB_Loss", g_loss[4], step=step)
                tf.summary.scalar("identityA_Loss", g_loss[5], step=step)
                tf.summary.scalar("identityB_Loss", g_loss[6], step=step)
                tf.summary.scalar("Total Loss", g_loss[0], step=step)
                
            if tensorboard_images_freq:
                if step % tensorboard_images_freq == 0:
                    # Generate images
                    test_input_A = tf.image.resize(next(testA), size=(cfg['height'], cfg['width']))
                    test_input_B = tf.image.resize(next(testB), size=(cfg['height'], cfg['width']))
                    fakeB = genA2B(test_input_A)
                    fakeA = genB2A(test_input_B)

                    # Get reconstructed images
                    reconsA = genB2A(fakeB)
                    reconsB = genA2B(fakeA)

                    identityA = genB2A(test_input_A)
                    identityB = genA2B(test_input_B)

                    tf.summary.image('fakeB', fakeB, max_outputs=8, step=step)
                    tf.summary.image('fakeA', fakeA, max_outputs=8, step=step)
                    tf.summary.image('reconsA', reconsA, max_outputs=8, step=step)
                    tf.summary.image('reconsB', reconsB, max_outputs=8, step=step)
                    tf.summary.image('identityA', identityA, max_outputs=8, step=step)
                    tf.summary.image('identityB', identityB, max_outputs=8, step=step)
                    tf.summary.image('test_input_A', test_input_A, max_outputs=8, step=step)
                    tf.summary.image('test_input_B', test_input_B, max_outputs=8, step=step)


            if (step + 1) % save_steps_freq == 0:

                genA2B.save(os.path.join('weight', 'generatorAToB_temp_%d.h5' % (step + 1)))
                genB2A.save(os.path.join('weight', 'generatorBToA_temp_%d.h5' % (step + 1)))
                discA.save(os.path.join('weight', 'discriminatorA_temp_%d.h5' % (step + 1)))
                discB.save(os.path.join('weight', 'discriminatorB_temp_%d.h5' % (step + 1)))
                



    print("Training completed. Saving weights.")
    genA2B.save(os.path.join('weight', 'generatorAToB.h5'))
    genB2A.save(os.path.join('weight', 'generatorBToA.h5'))
    discA.save(os.path.join('weight', 'discriminatorA.h5'))
    discB.save(os.path.join('weight', 'discriminatorB.h5'))


    
