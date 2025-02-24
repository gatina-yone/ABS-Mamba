import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import CreateDataLoader
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
from models.mamba_one import Mamba_model
from models.modules import SAM2Encoder

def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

if __name__ == '__main__':

    opt = TrainOptions().parse()

    opt.phase = 'train'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f'Training set size: {len(dataset)}')

    opt.phase = 'val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    print(f'Validation set size: {len(dataset_val)}')

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger, f"Experiment Name: {opt.name}")

    L1_avg = np.zeros([opt.niter + opt.niter_decay, len(dataset_val)])

    model = Mamba_model()
    model.initialize(opt)


    target_modules = ["layers.0", "layers.1", "attn.qkv"]
    peft_config = LoraConfig(
        inference_mode=False,
        r=16, 
        lora_alpha=32,  
        lora_dropout=0.1, 
        target_modules=target_modules,
        bias="none",  
        task_type="FEATURE_EXTRACTION" 
    )
    model.encoder = SAM2Encoder().get_encoder()
    model.encoder = get_peft_model(model.encoder, peft_config)
    print("LoRA configuration applied to the model.")


    trainable_params = []
    all_param = 0
    trainable_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params.append(name)
            trainable_param += param.numel()

    if trainable_params:
        print(f"trainable params: {trainable_param:,d} || "
              f"all params: {all_param:,d} || "
              f"trainable%: {100 * trainable_param / all_param:.2f}%")
    else:
        print("No parameters set as trainable! Check your LoRA configuration.")
        exit()

    loraoptimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=bnb.optim.Adam8bit,
        lr=1e-4,            
        loraplus_lr_ratio=8,
        weight_decay=0.01  
    )
    print("Optimizer successfully initialized.")

    scheduler = CosineAnnealingLR(
        loraoptimizer,
        T_max=opt.niter + opt.niter_decay,

        eta_min=1.0e-7
    )
    print("Learning rate scheduler successfully initialized.")

    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        print(f"Starting epoch {epoch} / {opt.niter + opt.niter_decay}")
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.8f}")
        opt.phase = 'train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, 0)

        if epoch % opt.save_epoch_freq == 0:
            print(f'Running validation for epoch {epoch}')
            opt.phase = 'val'
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')

            for i, data_val in enumerate(dataset_val):
                model.set_input(data_val)
                model.test()
                fake_im = model.fake_B.cpu().data.numpy()
                real_im = model.real_B.cpu().data.numpy()
                real_im = real_im * 0.5 + 0.5
                fake_im = fake_im * 0.5 + 0.5

                if real_im.max() > 0:
                    L1_avg[epoch - 1, i] = abs(fake_im - real_im).mean()

            l1_avg_loss = np.mean(L1_avg[epoch - 1])
            print_log(logger, f'Epoch {epoch:3d}   L1 Average Loss: {l1_avg_loss:.5f}')
            logger.close()
            print(f'Saving the model at the end of epoch {epoch}, total_steps {total_steps}')
            model.save('latest')
            model.save(epoch)
        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec')
        scheduler.step()
        print("Learning rate updated.")
    logger.close()
    print("Training completed successfully.")

