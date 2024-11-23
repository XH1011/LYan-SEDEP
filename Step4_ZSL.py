import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from trainer import Trainer
from datautils import ZSLDataset
import pandas as pd

all_results = []
results = []

for seed in range(38,39):
    print('seed:', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize results container for best results
    best_results = { 'acc': 0.0, 'y_real': None, 'y_pred': None}

    def parse_args() :
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='PU')
        # parser.add_argument('--dataset', type=str, default='chopper')
        parser.add_argument('--n_epochs', type=int, default=100)  # 修改前100
        parser.add_argument('--batch_size', type=int, default=50)  # 修改前50
        parser.add_argument('--latent_dim', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--gzsl', action='store_true', default=False)
        parser.add_argument('--da', action='store_true', default=True)  # 修改前default=False
        parser.add_argument('--ca', action='store_true', default=True)  # 修改前default=False
        parser.add_argument('--support', action='store_true', default=True)  # 修改前default=False
        parser.add_argument('--resume', action='store_true', default=False)  #
        return parser.parse_args()


    def main():
        # setup parameters for trainer
        args = parse_args()
        if args.dataset == 'chopper':
            x_dim = 128
            attr_dim = 30
            n_train = 27
            n_test = 3
        elif args.dataset == 'PU':
            x_dim = 128
            attr_dim = 30
            n_train = 27
            n_test = 3
        else:
            raise NotImplementedError

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0,
        }

        train_dataset = ZSLDataset(args.dataset, n_train, n_test, train=True, gzsl=args.gzsl)
        # dataset:(list(10320),2048)  #gzsl_dataset:(list(129000=645x200),2048)
        train_generator = DataLoader(train_dataset, **params)
        # {DataLoader:207} 以dataset:10320算的(10320/50)

        layer_sizes = {
            'x_enc': 400,
            'x_dec': 450,
            'c_enc': 350,
            'c_dec': 150
        }

        kwargs = {
            'gzsl': args.gzsl,
            'use_da': args.da,
            'use_ca': args.ca,
            'use_support': args.support,
        }

        train_agent = Trainer(
            device, args.dataset, x_dim, attr_dim, args.latent_dim,
            n_train, n_test, args.lr, layer_sizes, **kwargs
        )

        # load previous models, if any
        if args.resume == True:
            vae_start_ep = train_agent.load_models()  # 1.Load the trained parameters of the model
            # 加载之前已保存的模型参数，并将加载的模型的 epoch 数量赋值给 vae_start_ep
        else:
            vae_start_ep = 0
        # vae_start_ep = train_agent.load_models() ## 1.Load the trained parameters of the model #0

        print('Training the VAE')
        for ep in range(vae_start_ep + 1, args.n_epochs + 1):
            # train the VAE
            vae_loss = 0.0
            da_loss, ca_loss = 0.0, 0.0

            print("Train MVAE for the {} th epoch".format(ep))
            for idx, (img_features, attr, label_idx) in enumerate(train_generator):
                losses = train_agent.fit_VAE(img_features, attr, label_idx, ep)

                vae_loss += losses[0]
                da_loss += losses[1]
                ca_loss += losses[2]

            n_batches = idx + 1
            print("[VAE Training] Losses for epoch: [%3d] : %.4f(V), %.4f(D), %.4f(C)"
                  % (ep, vae_loss / n_batches, da_loss / n_batches, ca_loss / n_batches))

            # save VAE after each epoch
            train_agent.save_VAE(ep)

        seen_dataset = None
        if args.gzsl:
            seen_dataset = train_dataset.gzsl_dataset

        syn_dataset = train_agent.create_syn_dataset(
            train_dataset.test_classmap, train_dataset.attributes, seen_dataset)  # (LIST(28800=72X400),64)
        final_dataset = ZSLDataset(args.dataset, n_train, n_test,
                                   train=True, gzsl=args.gzsl, synthetic=True, syn_dataset=syn_dataset)
        final_train_generator = DataLoader(final_dataset, **params)
        # 576 (syn_dataset:28800/50)

        # compute accuracy on test dataset
        test_dataset = ZSLDataset(args.dataset, n_train, n_test, False, args.gzsl)  # (1440,2048)
        test_generator = DataLoader(test_dataset, **params)
        # 29 (test_dataset:test_unseen_loc:1440/50)

        best_acc = 0.0
        for ep in range(1, 1 + 2 * args.n_epochs):
            # train final classifier
            total_loss = 0
            for idx, (features, _, label_idx) in enumerate(final_train_generator):
                loss = train_agent.fit_final_classifier(features, label_idx)
                total_loss += loss

            total_loss = total_loss / (idx + 1)
            # print('[Final Classifier Training] Loss for epoch: [%3d]: %.3f' % (ep, total_loss))

            ## find accuracy on test data
            if args.gzsl:
                acc_s, acc_u = train_agent.compute_accuracy(test_generator)
                acc = 2 * acc_s * acc_u / (acc_s + acc_u)
                # print(acc, acc_s, acc_u)
            else:
                acc,y_real, y_pred = train_agent.compute_accuracy(test_generator)

            if acc > best_results['acc']:
                best_results.update({'acc': acc, 'y_real': y_real, 'y_pred': y_pred})
                if args.gzsl:
                    best_acc_s = acc_s
                    best_acc_u = acc_u

        if args.gzsl:
            print('Best Accuracy: %.3f ==== Seen: [%.3f] -- Unseen[%.3f]' % (best_acc, best_acc_s, best_acc_u))
        else:
            print('Best Accuracy: %.3f' % best_results['acc'])
        results.append({'seed': seed, 'ACC': best_results['acc'], 'y_real': best_results['y_real'],
                        'y_pred': best_results['y_pred']})
        df = pd.DataFrame(results)
        # df.to_excel('./ZSL_results/results_pu.xlsx', sheet_name='1', index=False)
        # df.to_excel('./ZSL_results/results_chopper.xlsx', sheet_name='1', index=False)

    if __name__ == '__main__':
        main()