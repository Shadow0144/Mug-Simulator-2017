/******************************************************************************
* Coherent Point Drift
* Copyright (C) 2014 Pete Gadomski <pete.gadomski@gmail.com>
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
******************************************************************************/

#include "cpd/affinity_matrix.hpp"


namespace cpd
{


void construct_affinity_matrix(const arma::mat& X, const arma::mat& Y,
                               float beta, arma::mat& G)
{
    arma::mat Xr = X.cols(0, 2);
    arma::mat Yr = Y.cols(0, 2);
    double k = -2 * std::pow(beta, 2);
    const arma::uword N = Xr.n_rows;
    const arma::uword M = Yr.n_rows;
    const arma::uword D = Yr.n_cols;

    G.set_size(N, M);

    for (arma::uword i = 0; i < M; ++i)
    {
        G.col(i) = arma::exp(arma::sum(arma::pow(Xr - arma::repmat(Yr.row(i), N, 1), 2),
                                       1) / k);
    }

}


}
