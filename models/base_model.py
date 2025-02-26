import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_frequency_modes(seq_len, modes=4, mode_select_method='random'):
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierBlock(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(node, in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q):
        B, D, N, L = q.shape
        x = q.permute(0, 2, 1, 3)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, N, D, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])

        output = torch.fft.irfft(out_ft, n=x.size(-1)).permute(0, 2, 1, 3)
        return (output, None)


class FourierCrossAttention(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len_q, seq_len_kv,
                 modes=64, mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print('Fourier enhanced cross attention is used.')
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask=None):
        B, E, H, L = q.shape
        xq = q.permute(0, 2, 1, 3) # size = [B, H, E, L]
        xk = k.permute(0, 2, 1, 3)
        xv = v.permute(0, 2, 1, 3)

        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        output = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 2, 1, 3)
        return (output, None)


class FourierSqeAttentionLayer(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64,
                 mode_select_method='random', activation='tanh', policy=0):
        super(FourierSqeAttentionLayer, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask=None):
        B, E, H, L = q.shape
        xq = q.permute(0, 2, 1, 3)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 1, 3)
        xv = v.permute(0, 2, 1, 3)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))

        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)

        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        output = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 2, 1, 3)

        return (output, None)


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, Fourier_option, attention_option, modes, activation,
                 order=4, non_linear='linear', d_ff=None, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(torch.rand(1, order, 1, time_step, self.multi * self.time_step).cuda())
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        self.modes = modes
        self.non_linear = non_linear
        self.activation = activation
        self.attopt = attention_option
        if Fourier_option == 'FB':
            self.Fourier = FourierBlock(node=self.unit, in_channels=order, out_channels=order, seq_len=time_step, modes=self.modes, mode_select_method='random')
        # if Fourier_option == 'FCA':
        #     self.Fourier = FourierCrossAttention(node=self.unit, in_channels=order, out_channels=order,
        #                                          seq_len_q=time_step, seq_len_kv=2, modes=self.modes,
        #                                          mode_select_method='random')
        self.dropout = nn.Dropout(0.1)
        moving_avg = [2]
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        # t_embed = 4 * self.unit
        # out = 8
        # if self.attopt == 'fourier':
        #     self.feedforward = nn.Linear(t_embed, t_embed)
        #     self.linearq = nn.Linear(t_embed, out)
        #     self.lineark = nn.Linear(t_embed, out)
        #     self.linearv = nn.Linear(t_embed, out)
        #     self.attentionlayer = FourierSqeAttentionLayer(node=self.unit, in_channels=4, out_channels=4,
        #                                                    seq_len_q=time_step, seq_len_kv=time_step, modes=self.modes,
        #                                                    mode_select_method='random', activation=self.activation)
        #     self.lineart = nn.Linear(t_embed, t_embed)

    def compl_mul1d(self, input, weights):
        return torch.einsum("bni,nio->bno", input, weights)

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        new_x, _ = self.Fourier(input)
        x = input + self.dropout(new_x)
        xt = x.view(batch_size, time_step, -1)
        x_s, x_t = self.decomp1(xt)

        # if self.attopt == 'fourier':
        #     x_s_o = x_s
        #     x_s = self.dropout(x_s.reshape(batch_size, -1, node_cnt, time_step))
        #     x_s_a, _ = self.attentionlayer(input, x_s, x_s)
        #     x_s_a = x_s_a.reshape(batch_size, time_step, -1)
        #     x_s = x_s_o + x_s_a
        #     x = x_s + x_t

        if self.attopt == 'linear':
            x = x_s + x_t
        x = x.reshape(batch_size, -1, node_cnt, time_step)
        return x

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        gfted = torch.matmul(gfted, self.weight)
        igfted = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.sum(igfted, dim=1)
        if self.non_linear == 'linear':
            forecast_source = self.forecast(igfted).squeeze(1)
        elif self.non_linear == 'sigmoid':
            forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        elif self.non_linear == 'relu':
            forecast_source = torch.relu(self.forecast(igfted).squeeze(1))
        elif self.non_linear == 'tanh':
            forecast_source = torch.tanh(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        backcast_short = self.backcast_short_cut(x).squeeze(1)
        backcast_source = torch.sigmoid(backcast_short - self.backcast(igfted))
        x_back = torch.sigmoid(self.backcast(igfted))
        return forecast, backcast_source, x_back


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 coe_a=1, coe_b=1, order=4, gconv='gegen', non_linear='linear', Fouropt='FB', attention_set='time',
                 modes=4, activation='softmax', device='cpu'):
        super(Model, self).__init__()
        self.unit = units
        self.non_linear = non_linear
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.Fouropt = Fouropt
        self.attention_option = attention_set
        self.modes = modes
        self.order = order
        self.activation = activation
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, order=self.order, non_linear= self.non_linear,
                             Fourier_option=self.Fouropt, attention_option=self.attention_option, modes=self.modes,
                             activation=self.activation, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc_1 = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), int(self.time_step)),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gconv = gconv
        self.coe_a = coe_a
        self.coe_b = coe_b
        self.to(device)

    def get_laplacian(self, graph, normalize):

        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    # def Mono_polynomial(self, laplacian):
    #     N = laplacian.size(0)
    #     laplacian = laplacian.unsqueeze(0)
    #     first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
    #     second_laplacian = first_laplacian - laplacian
    #     third_laplacian = torch.matmul(second_laplacian, second_laplacian)
    #     multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)
    #     return multi_order_laplacian
    #
    # def Bern_polynomial(self, laplacian):
    #     N = laplacian.size(0)
    #     laplacian = laplacian.unsqueeze(0)/2
    #     one_lapla = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
    #     laplacian_1 = one_lapla - laplacian
    #     first_laplacian = torch.matmul(laplacian_1, laplacian_1)
    #     second_laplacian = 2 * torch.matmul(laplacian, laplacian_1)
    #     third_laplacian = 2 * torch.matmul(laplacian, laplacian)
    #     multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)
    #     return multi_order_laplacian

    def Cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = 2 * torch.matmul(laplacian, second_laplacian) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def Jacobi_coe(self, k, a=1, b=1, l=-1.0, r=1.0):
        theta_0 = (2 * k + a + b) * (2 * k + a + b - 1) / (2 * k * (k + a + b))
        theta_1 = (2 * k + a + b - 1) * (a * a - b * b) / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        theta_2 = (k + a - 1) * (k + b - 1) * (2 * k + a + b) / (k * (k + a + b) * (2 * k + a + b - 2))
        return [theta_0, theta_1, theta_2]

    def Jacobi_polynomial(self, laplacian, a=1, b=1, l=-1.0, r=1.0):

        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = (a - b) / 2 + (a + b + 2) / 2 * laplacian
        Theta_2 = self.Jacobi_coe(2, a, b)
        third_laplacian = torch.matmul((Theta_2[0] * laplacian + Theta_2[1]), second_laplacian) \
                          + Theta_2[2] * first_laplacian
        Theta_3 = self.Jacobi_coe(3, a, b)
        forth_laplacian = torch.matmul((Theta_3[0] * laplacian + Theta_3[1]), third_laplacian) \
                          + Theta_3[2] * second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def Gegen_coe(self, k, a=1, l=-1.0, r=1.0):
        theta_0 = 2 * (k + a - 1)
        theta_1 = k + 2 * (a - 1)
        return [theta_0, theta_1]

    def Gegen_polynomial(self, laplacian, a=1, l=-1.0, r=1):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
        k = r
        if k == 1:
            multi_order_laplacian = first_laplacian
        if k == 2:
            second_laplacian = 2 * a * laplacian
            multi_order_laplacian = torch.cat([first_laplacian, second_laplacian], dim=0)
        if k == 3:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = 1 / 2 * (torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                                       - Theta_2[1] * first_laplacian)
            multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)
        if k == 4:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = 1 / 2 * (torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                                       - Theta_2[1] * first_laplacian)
            Theta_3 = self.Gegen_coe(3, a)
            forth_laplacian = 1 / 3 * (torch.matmul((Theta_3[0] * laplacian), third_laplacian)
                                       - Theta_3[1] * second_laplacian)
            multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian,
                                               forth_laplacian], dim=0)
        if k == 5:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = 1 / 2 * (torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                                       - Theta_2[1] * first_laplacian)
            Theta_3 = self.Gegen_coe(3, a)
            forth_laplacian = 1 / 3 * (torch.matmul((Theta_3[0] * laplacian), third_laplacian)
                                       - Theta_3[1] * second_laplacian)
            Theta_4 = self.Gegen_coe(4, a)
            fifth_laplacian = 1 / 3 * (torch.matmul((Theta_4[0] * laplacian), forth_laplacian)
                                       - Theta_3[1] * third_laplacian)
            multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian,
                                               fifth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat))

        if self.gconv == 'cheby':
            mul_L = self.Cheb_polynomial(laplacian)
        elif self.gconv == 'jacobi':
            mul_L = self.Jacobi_polynomial(laplacian, self.coe_a, self.coe_b)
        elif self.gconv == 'gegen':
            mul_L = self.Gegen_polynomial(laplacian, self.coe_a, r=self.order)
        # elif self.gconv == 'bern':
        #     mul_L = self.Bern_polynomial(laplacian)
        # elif self.gconv == 'mono':
        #     mul_L = self.Mono_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        back = []
        for stack_i in range(self.stack_cnt):
            forecast, X, x_back = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
            back.append(x_back)
        forecast = torch.stack(result).sum(0)
        foreback = torch.stack(back).sum(0)
        forecast = self.fc_1(forecast)
        foreback = self.fc_2(foreback)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention, foreback.squeeze(1)
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention, foreback.squeeze(1).permute(0, 2, 1).contiguous()
