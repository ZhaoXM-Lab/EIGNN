import os
import pandas as pd
import torch
from torch import nn
import numpy as np
import networkx as nx
from functools import partial, reduce
from einops import rearrange
from einops.layers.torch import Rearrange
import gzip


class NoScaleDropout(nn.Module):
    """
        Dropout without rescaling and variable dropout rates.
        https://github.com/pytorch/pytorch/issues/7544
    """

    def __init__(self, rate_max) -> None:
        super().__init__()
        self.rate_max = rate_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate_max == 0:
            return x
        else:
            rate = torch.empty(1, device=x.device).uniform_(0, self.rate_max)
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - rate)
            return x * mask


class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, x):
        return self.fn(x) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear, norm=nn.BatchNorm1d):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        # nn.BatchNorm1d(dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),

        dense(dim * expansion_factor, dim),
        norm(dim),
        nn.Dropout(dropout)
    )


class SNP2GENE(nn.Module):
    def __init__(self, eqtls, useeqtl=True, embed_size=1):
        super().__init__()

        eqtl_snp = eqtls['snp'].values
        eqtl_gene = eqtls['gene'].values

        snp_list = eqtls['snp'].drop_duplicates()
        gene_list = eqtls['gene'].drop_duplicates()

        num_snp = len(snp_list)
        num_gene = len(gene_list)
        gene_size = 1

        if useeqtl:
            snp_mask = pd.DataFrame(np.zeros([num_snp, num_gene * embed_size]),
                                    index=snp_list,
                                    columns=gene_list.repeat(embed_size))
            inx_snp = [snp_mask.index.get_loc(i) for i in eqtl_snp]
            inx_gene = [np.where(snp_mask.columns == i)[0] for i in eqtl_gene]
            snp_mask = snp_mask.values.astype(np.int8)
            for i, j in zip(inx_snp, inx_gene):
                snp_mask[i, j] = 1
            snp_mask = torch.tensor(snp_mask, dtype=torch.int8)

        else:
            snp_mask = torch.ones(num_snp, num_gene * embed_size, dtype=torch.int8)

        self.snp_weight = torch.nn.Parameter(
            data=torch.ones(num_snp, num_gene * embed_size) / snp_mask.sum(dim=0, keepdim=True),
            requires_grad=True)
        self.snp_bias = torch.nn.Parameter(data=torch.zeros(num_gene * embed_size),
                                           requires_grad=True)
        self.snp_mask = torch.nn.Parameter(data=snp_mask,
                                           requires_grad=False)
        self.bn_snp = torch.nn.BatchNorm1d(num_gene * embed_size)

        gene_mask = torch.zeros(num_gene * embed_size, num_gene * gene_size, dtype=torch.int8)
        for i in range(gene_mask.shape[1]):
            gene_mask[i * embed_size:(i + 1) * embed_size, i * gene_size:(i + 1) * gene_size] = 1

        self.gene_weight = torch.nn.Parameter(
            data=torch.ones(num_gene * embed_size, num_gene * gene_size) / gene_mask.sum(dim=0, keepdim=True),
            requires_grad=True)
        self.gene_bias = torch.nn.Parameter(data=torch.zeros(num_gene * gene_size),
                                            requires_grad=True)
        self.gene_mask = torch.nn.Parameter(data=gene_mask,
                                            requires_grad=False)
        self.bn_gene = torch.nn.BatchNorm1d(num_gene)

        self.snp_list = snp_list
        self.gene_list = gene_list.repeat(gene_size)

        self.ext_layer = nn.Sequential(nn.Linear(num_gene * embed_size, num_gene * embed_size),
                                       nn.BatchNorm1d(num_gene * embed_size),
                                       nn.GELU(),
                                       )

    def forward(self, x):
        gene_feat = torch.mm(x, self.snp_weight * self.snp_mask) + self.snp_bias
        gene_feat = self.bn_snp(gene_feat)
        # gene_feat = self.ext_layer(gene_feat)
        gene_in = nn.functional.gelu(gene_feat)

        gene = torch.mm(gene_in, self.gene_weight * self.gene_mask) + self.gene_bias  # increase gene dim not help
        # gene = self.bn_gene(gene) # bn_gene hurt performance
        gene = nn.functional.gelu(gene)

        return gene_feat, gene


class GENE_NET(nn.Module):
    def __init__(self, go_graph_path, go_anno_path, input_gene_list: list, num_hiddens_genotype, final_dim, dropout,
                 final_depth=1, final_expansion_factor=4, n_class=1, gene_side_path=True):
        '''

        Args:
            go_graph_path: path for go graph define by netowrkx.DiGraph
            go_anno_path: annotations for each term of the go graph
            num_hiddens_genotype:
            input_gene_list: the gene list of the input (ensemble id)
        '''

        super().__init__()
        self.gene_side_path = gene_side_path
        input_gene_list = np.array(input_gene_list)

        dG = nx.read_adjlist(go_graph_path, create_using=nx.DiGraph)
        self.root = list(filter(lambda x: dG.in_degree(x) < 1, dG.nodes))
        assert len(self.root) >= 1

        anno = pd.read_csv(go_anno_path, index_col=0)

        term_direct_gene_map = dict()
        num_gene_of_term = dict()

        if 'GO_ID' in anno.columns:
            for term in anno['GO_ID'].drop_duplicates().values:
                genes = anno[anno['GO_ID'] == term]['ENSG'].values.tolist()
                term_direct_gene_map[term] = list(reduce(lambda x, y: x + y,
                                                         [np.where(input_gene_list == i)[0].tolist() for i in genes]))
                num_gene_of_term[term] = len(term_direct_gene_map[term])

        # reactome graph
        else:
            for term in anno['pathway'].drop_duplicates().values:
                genes = anno[anno['pathway'] == term]['source'].values.tolist()
                term_direct_gene_map[term] = list(reduce(lambda x, y: x + y,
                                                         [np.where(input_gene_list == i)[0].tolist() for i in genes]))
                num_gene_of_term[term] = len(term_direct_gene_map[term])

        self.term_direct_gene_map = term_direct_gene_map
        self.go_net, self.term_layer_list, self.term_neighbor_map = self.construct_NN_graph(dG,
                                                                                            num_gene_of_term,
                                                                                            num_hiddens_genotype,
                                                                                            n_class)

        # add modules for final layer
        final_in = 0
        for node in self.root:
            final_in += self.go_net[node + '_layer'][0].dim

        self.final_layer = nn.Sequential(
            *[Residual(final_in, FeedForward(final_in, final_expansion_factor, dropout=dropout))
              for _ in range(final_depth)],
        )

        self.final_aux = nn.Sequential(nn.Linear(final_in, final_dim),
                                       # nn.Dropout(dropout),
                                       # nn.ReLU() # seems like adding this decrease the performance
                                       )

        data_dir = '/data/home/zhangzc/dataset/brain_imgen/'
        with gzip.open(os.path.join(data_dir, 'GGI', '9606.protein.links.v12.0.txt.gz'),
                       'r') as f:
            ppi = pd.read_csv(f, sep=' ', header=0, index_col=None)
        gene2pro = pd.read_csv(os.path.join(data_dir, 'GGI', 'gene2pro.csv'))

        ppi['protein1'] = ppi['protein1'].str.replace('9606.', '', regex=False)
        ppi['protein2'] = ppi['protein2'].str.replace('9606.', '', regex=False)
        gene2pro = gene2pro[gene2pro['gene'].isin(input_gene_list)]
        ppi = ppi[ppi['protein1'].isin(gene2pro['protein'].values) &
                  ppi['protein2'].isin(gene2pro['protein'].values)]
        p2g = {}
        for g, p in gene2pro.values:
            p2g[p] = g

        ggi = ppi.copy()
        ggi.columns = ['gene1', 'gene2', 'combined_score']
        ggi['gene1'] = ggi['gene1'].apply(lambda x: p2g[x])
        ggi['gene2'] = ggi['gene2'].apply(lambda x: p2g[x])

        mask = np.zeros([len(input_gene_list), len(input_gene_list)])
        i1 = np.where(ggi['gene1'].values.reshape(-1, 1) == input_gene_list)[1]
        i2 = np.where(ggi['gene2'].values.reshape(-1, 1) == input_gene_list)[1]
        mask[i1, i2] = 1
        if gene_side_path:
            self.connection = GeneSidePath(n_token=len(input_gene_list), token_dim=1, out_dim=final_in,
                                           depth=2, heads=8, dim_head=8, dropout=0., mask=mask)

    def forward(self, x):
        '''

        Args:
            x:

        Returns:

        '''
        term_gene_out_map = dict()
        for term, gene_inx in self.term_direct_gene_map.items():
            term_gene_out_map[term] = x[..., gene_inx]

        term_NN_out_map = {}
        aux_out_map = {}
        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)

                term_NN_out_map[term] = self.go_net[term + '_layer'](child_input)
                aux_out_map[term] = self.go_net[term + '_aux_layer'](term_NN_out_map[term])

        final_input = torch.cat([term_NN_out_map[node] for node in self.root], dim=-1)

        term_NN_out_map['final'] = self.final_layer(final_input)
        # final = self.final_aux(term_NN_out_map['final'])
        if self.gene_side_path:
            final = self.final_aux(term_NN_out_map['final'] + self.connection(x))
        else:
            final = self.final_aux(term_NN_out_map['final'])

        # return aux_out_map
        return final, list(aux_out_map.values())

    @staticmethod
    def construct_NN_graph(dG, num_gene_of_term, num_hiddens_genotype, n_class):
        term_layer_list = []  # term_layer_list stores the built neural network
        term_neighbor_map = {}

        go_net = nn.ModuleDict()

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            if len(leaves) < 1:
                break
            term_layer_list.append(leaves)

            for term in leaves:
                # input size will be #chilren + #genes directly annotated by the term
                input_size = sum([go_net[child + '_layer'][0].dim for child in term_neighbor_map[term]])
                if term in num_gene_of_term:
                    input_size += num_gene_of_term[term]

                # num_hiddens_genotype is the number of the hidden variables in each state
                layer_depth = 2
                go_net[term + '_layer'] = nn.Sequential(*[Residual(input_size, FeedForward(input_size,
                                                                                           4,
                                                                                           dropout=0.1))
                                                          for _ in range(layer_depth)])
                go_net[term + '_aux_layer'] = nn.Sequential(
                    # nn.BatchNorm1d(input_size),
                    nn.Linear(input_size, n_class),
                    # nn.Sigmoid()
                )
            dG.remove_nodes_from(leaves)
        return go_net, term_layer_list, term_neighbor_map


class PatchEmbedding(nn.Module):
    '''Embedding for MRI patches (modified from LDMIL)
    '''

    def __init__(self, patch_size, inplanes, embed_size=4, stride=1, downsample=None):
        super(PatchEmbedding, self).__init__()

        self.outsize = embed_size
        patch_size = np.array(patch_size)

        self.conv1 = nn.Conv3d(inplanes, 16, kernel_size=5, stride=1, padding=2)
        self.norm1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm3d(32)

        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm3d(32)

        self.relu = nn.GELU()
        self.maxpool3D = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        fc_insize = int(np.prod(patch_size // 2 // 2 // 2)) * 32
        # fc_insize = 32
        self.fc7 = nn.Sequential(nn.Linear(fc_insize, 32),
                                 nn.GELU())
        self.drop = nn.Dropout(0.3)
        self.fc8 = nn.Sequential(nn.Linear(32, self.outsize),
                                 nn.GELU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.maxpool3D(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)
        x = self.maxpool3D(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.relu(x)
        x = self.maxpool3D(x)
        x = x.view(x.shape[0], -1)

        x = self.fc7(x)
        x = self.drop(x)
        x = self.fc8(x)

        return x


class CrossAtt(nn.Module):
    def __init__(self, gene_feat_dim, img_feat_dim, dim_head, heads=1, dropout=0.):
        super(CrossAtt, self).__init__()
        # head=1
        inner_dim = dim_head * heads

        self.gene_feat_dim = gene_feat_dim
        self.img_feat_dim = img_feat_dim
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.heads = heads

        self.to_q = nn.Linear(gene_feat_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(img_feat_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(img_feat_dim, inner_dim, bias=False)
        self.scale = dim_head ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, img_feat_dim),
            nn.Dropout(dropout)
        )

    def forward(self, gene, img):
        '''

        Args:
            gene: b x ng x d
            img: b x ni x d

        Returns:

        '''
        gene = gene.view(gene.shape[0], -1, self.gene_feat_dim)

        q = self.to_q(gene)
        k = self.to_k(img)
        v = self.to_v(img)  # b x ni x (head x dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b x h x ng x ni
        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)  # b x h x ng x d
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out).view(out.shape[0], -1)
        return out


class BatchAtt(nn.Module):
    def __init__(self, x_dim, y_dim, feat_dim, momentum=0.9):
        super(BatchAtt, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.feat_dim = feat_dim

        att = torch.ones(x_dim, y_dim)
        # self.query = nn.Linear(feat_dim, feat_dim)
        self.register_buffer('att', torch.softmax(att, dim=-1),
                             persistent=True)

        self.alpha = momentum
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, y):
        feat = y

        if self.training:
            eps = 1e-6
            # x = self.query(rearrange(x, 'n (g d) -> n g d', d=self.feat_dim))

            x = rearrange(x, 'n (g d) -> g (n d)', d=self.feat_dim)
            y = rearrange(y, 'n p d -> p (n d)')
            y = y / (y.norm(p=2, dim=1, keepdim=True) + eps)
            x = x / (x.norm(p=2, dim=1, keepdim=True) + eps)
            att = torch.softmax(x.mm(y.t()), dim=-1)
            self.att = self.att * self.alpha + (1 - self.alpha) * att.detach()
        else:
            att = self.att

        feat = torch.matmul(att, feat)
        feat = rearrange(feat, 'n g d -> n (g d)')
        feat = self.dropout(feat)
        return feat


class MaskedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., mask=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if mask is None:
            self.mask = 1
        else:
            self.mask = torch.nn.Parameter(data=mask,
                                           requires_grad=False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots * self.mask

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GeneSidePath(nn.Module):
    def __init__(self, n_token, token_dim, out_dim, depth, heads, dim_head, dropout=0., mask=None):
        super().__init__()
        self.n_token = n_token
        self.token_dim = token_dim

        self.layers = nn.ModuleList([])
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MaskedAttention(token_dim, heads=heads, dim_head=dim_head, dropout=dropout, mask=mask),
                FeedForward(token_dim, expansion_factor=4, dropout=dropout,
                            norm=lambda d: nn.Sequential(Rearrange('b nt d -> b d nt'),
                                                         nn.BatchNorm1d(d),
                                                         Rearrange('b d nt -> b nt d')))
            ]))

        self.out = nn.Linear(n_token * token_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.n_token, self.token_dim)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.view(-1, self.n_token * self.token_dim)
        x = self.out(x)
        return x


class EIGNN(nn.Module):  # Explainable imaging genetics Network
    def __init__(self, num_patches, patch_size, channels, img_dim, depth,
                 eqtl_path, graph_dir, dropout=0., snp_dropout=0., gene_dropout=0., num_classes=1, embed_size=6,
                 useimg=True, usesnp=True, useeqtl=True, usegraph=True, cross_int=True, train_gene=True,
                 lastdropout=True, only_train_last=False, fix_embed=False, gene_side_path=True):
        super().__init__()
        if num_classes > 1:
            raise NotImplementedError
        if cross_int:
            assert useimg and usesnp
        assert useimg or usesnp
        self.useimg = useimg
        self.usesnp = usesnp
        self.usegraph = usegraph
        self.cross_int = cross_int
        self.embed_size = embed_size

        ## genotype
        eqtls = pd.read_csv(eqtl_path, sep='\t')
        eqtls.columns = ['snp', 'gene']
        gene_dim = len(eqtls['gene'].drop_duplicates().values)  # number of genes

        self.snp2gene = SNP2GENE(eqtls, useeqtl, embed_size=embed_size)
        self.snp_drop = NoScaleDropout(snp_dropout)
        self.gene_drop = nn.Dropout(gene_dropout)

        eqtl_dir, eqtlfile = os.path.split(eqtl_path)
        tissue, _, pt, _ = eqtlfile.split('.')

        if 'go_graph' in graph_dir:
            go_graph_path = os.path.join(graph_dir, '.'.join([tissue, 'go_graph', pt, 'adj']))
            go_anno_path = os.path.join(graph_dir, '.'.join([tissue, 'go_anno', pt, 'csv']))
        else:
            go_graph_path = os.path.join(graph_dir, '.'.join([tissue, 'react_graph', pt, 'adj']))
            go_anno_path = os.path.join(graph_dir, '.'.join([tissue, 'react_anno', pt, 'csv']))

        if usegraph:
            self.gene_net = GENE_NET(go_graph_path, go_anno_path,
                                     input_gene_list=[i.split('|')[0] for i in self.snp2gene.gene_list.values],
                                     num_hiddens_genotype=embed_size, final_dim=img_dim, dropout=dropout,
                                     final_depth=depth, n_class=num_classes, gene_side_path=gene_side_path)
        else:
            expansion_factor = 4
            self.gene_net = nn.Sequential(
                *[Residual(gene_dim, FeedForward(gene_dim, expansion_factor, dropout=dropout))
                  for _ in range(depth)],
                nn.Linear(gene_dim, img_dim),
                nn.BatchNorm1d(img_dim),
            )

        ## image
        self.img_embed = nn.ModuleList()
        for i in range(num_patches):
            self.img_embed.append(PatchEmbedding(patch_size, inplanes=1, embed_size=embed_size))
        if cross_int:
            img_mlp_dim = embed_size * len(self.snp2gene.gene_list.drop_duplicates())
        else:
            img_mlp_dim = embed_size * num_patches
        self.img_mlp = nn.Sequential(nn.Linear(img_mlp_dim, img_mlp_dim),
                                     nn.BatchNorm1d(img_mlp_dim),
                                     nn.GELU(),
                                     nn.Dropout(0.3),

                                     nn.Linear(img_mlp_dim, img_dim),
                                     )

        out_dim = 0
        if useimg:
            out_dim += img_dim
        if usesnp:
            out_dim += img_dim

        if lastdropout:
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(out_dim, num_classes),
                # nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(out_dim, num_classes),
                # nn.Sigmoid()
            )
        # self.batatt = BatchAtt(x_dim=gene_dim, y_dim=num_patches, feat_dim=embed_size)
        self.cratt = CrossAtt(gene_feat_dim=embed_size, img_feat_dim=embed_size,
                              dim_head=embed_size * 8, heads=8, dropout=0.)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = focal_loss(alpha=torch.tensor([.75, .25]), gamma=2, reduction='mean', )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if not train_gene:
            for param in self.snp2gene.children():
                param.requires_grad_(False)
            for param in self.gene_net.children():
                param.requires_grad_(False)

        if only_train_last:
            for param in self.children():
                param.requires_grad_(False)
            for param in self.output.children():
                param.requires_grad_(True)
            for param in self.img_mlp.children():
                param.requires_grad_(True)
        if fix_embed:
            for param in self.snp2gene.children():
                param.requires_grad_(False)
            for param in self.gene_net.children():
                param.requires_grad_(False)
            for param in self.img_embed.children():
                param.requires_grad_(False)

    def img_embedding(self, x, cross_x=None):
        fea_list = []
        cossim = None
        eps = 1e-6
        for p in range(x.shape[1]):
            fea_list.append(self.img_embed[p](x[:, p, :, :, :, :]))

        if cross_x is not None:
            # gfeature = self.batatt(x=cross_x, y=torch.stack(fea_list, 1))
            gfeature = self.cratt(gene=cross_x, img=torch.stack(fea_list, 1))
        else:
            gfeature = torch.cat(fea_list, -1)
        out = self.img_mlp(gfeature)
        return out, fea_list, cossim

    def gene_embedding(self, x):
        aux = []
        x = self.snp_drop(x)
        gene_feat, gene = self.snp2gene(x)
        gene = self.gene_drop(gene)
        if self.usegraph:
            root_feat, aux = self.gene_net(gene)
        else:
            root_feat = self.gene_net(gene)
        return root_feat, gene_feat, aux

    def forward(self, img, snp, aux_out=False, std_out=False):
        x = []
        aux = None
        img_x = None
        genetic_x = None
        cross_int = None
        patch_feats = None
        gene_feat = None

        if self.usesnp:
            genetic_x, gene_feat, aux = self.gene_embedding(snp)
            x.append(genetic_x)
        if self.useimg:
            if self.useimg and self.usesnp and self.cross_int:
                img_x, patch_feats, cross_int = self.img_embedding(img, cross_x=gene_feat)
            else:
                img_x, patch_feats, cross_int = self.img_embedding(img)
            x.append(img_x)

        x = torch.cat(x, dim=-1)
        x = self.output(x)

        if aux_out:
            return x, aux, img_x, genetic_x, [patch_feats, gene_feat]
        elif std_out:
            return x
        else:
            return x,

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)
                outputs = self(inputs, aux_labels[:, 0])
                predicts.append(outputs)
                groundtruths.append(labels[:, 0, :])  # multi patch
                group_labels.append(dis_label)

            device = next(self.parameters()).device
            pred = [i[0] for i in predicts]
            pred = torch.cat(pred, 0)
            pred = torch.sigmoid(pred)
            groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype)
            group_labels = torch.cat(group_labels, dim=0).to(torch.long)
            val_loss = self.criterion(pred.to(device),
                                      groundtruths.to(device=device))

            pred = pred.unsqueeze(-1).cpu().numpy()
            groundtruths = groundtruths.unsqueeze(-1).cpu().numpy()
            group_labels = group_labels.cpu().numpy()
            val_loss = val_loss.cpu().item()
        return pred, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        self.train(True)
        losses = torch.zeros(1, dtype=dtype, device=device, )

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        aux_labels_buf = torch.Tensor()
        labels_buf = torch.Tensor()
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data
            ## to collect data for the case that input might contains nan
            inx = ~torch.isnan(labels.view(labels.shape[0], -1)[:, 0])
            if self.useimg:
                inx = inx & (~torch.isnan(inputs.view(inputs.shape[0], -1)[:, 0]))
            if self.usesnp:
                inx = inx & (~torch.isnan(aux_labels.view(aux_labels.shape[0], -1)[:, 0]))
            inputs_buf = torch.cat([inputs_buf, inputs[inx]], 0)
            aux_labels_buf = torch.cat([aux_labels_buf, aux_labels[inx]], 0)
            labels_buf = torch.cat([labels_buf, labels[inx]], 0)
            if (n + 1) < len(train_loader):
                if inputs_buf.shape[0] < batch_size + 2:  # batch norm must use more than 1 sample
                    continue
                else:
                    inputs = inputs_buf[:batch_size]
                    aux_labels = aux_labels_buf[:batch_size]
                    labels = labels_buf[:batch_size]

                    inputs_buf = inputs_buf[batch_size:]
                    aux_labels_buf = aux_labels_buf[batch_size:]
                    labels_buf = labels_buf[batch_size:]
            else:
                inputs = inputs_buf
                aux_labels = aux_labels_buf
                labels = labels_buf
            c += 1
            ##

            # multi patch
            labels = labels[:, 0, :].to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            inputs = inputs.to(device=device, dtype=dtype)

            optimizer.zero_grad()
            outputs, aux, img_x, gene_x, cross_int = self(inputs, aux_labels[:, 0], aux_out=True)

            assert labels.shape[1] == 1
            loss = self.criterion(outputs, labels[:, 0, :])
            for aux_pre in (aux if aux is not None else []):
                loss += self.criterion(aux_pre, labels[:, 0, :]) / len(aux) * 0.05
            if self.useimg and self.usesnp:
                loss += -torch.cosine_similarity(img_x, gene_x, dim=1).mean() * 0.1

            loss.backward(retain_graph=True)
            losses += loss.detach()
            optimizer.step()
        return losses / c
