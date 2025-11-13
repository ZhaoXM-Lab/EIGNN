def attribution(dataloader_list, models, clfsetting, device):
    dtype = torch.float32
    attr = []
    for net, (train_loader, val_loader, test_loader) in zip(models, dataloader_list):
        net.train(False)
        net = net.to(device)
        exp = LayerIntegratedGradients(net, net.snp2gene)

        for i, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data
            inputs = inputs.to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            target = [0 for i in range(inputs.shape[0])]

            attribution = exp.attribute((inputs, aux_labels[:, 0]), additional_forward_args=(False, True),
                                        target=target, attribute_to_layer_input=True)
            attr.append(attribution.abs().mean(0).detach().cpu().numpy())
    attr = np.array(attr).mean(0)
    sorted_snp = net.snp2gene.snp_list.values[np.argsort(attr)[::-1]]

    return sorted_snp, np.sort(attr)[::-1]
