"""Second round clearing by Eisenburg Noe using the ClearingSystem class."""

import numpy as np


class ClearingSystem:
    """Represents a clearing system using the Eisenburg-Noe algorithm."""

    def __init__(self, external_assets, liability_matrix):
        """Initialize the clearing system with external assets and interbank liabilities."""
        self.external_assets = external_assets
        self.liability_matrix = liability_matrix
        self.num_banks = len(external_assets)
        self.total_liabilities = np.sum(liability_matrix, axis=1, dtype=float)
        self.payment_matrix = liability_matrix / self.total_liabilities[:, None]
        self.payment_matrix[np.isnan(self.payment_matrix)] = 0

    def calc_payments(self, calculation_method="Standard", max_iterations=500, tolerance=0.0001):
        """Calculate payments using specified method. Return payment vector."""
        if calculation_method == 'Standard':
            payments = self.total_liabilities.copy()
            previous_payments = np.zeros_like(payments)

            while np.any(payments != previous_payments):
                previous_payments = payments.copy()
                payments_received = np.matmul(self.payment_matrix.T, payments)
                defaulted_positions = (self.external_assets + payments_received <
                                       self.total_liabilities)
                payments[defaulted_positions] = np.minimum(
                    self.total_liabilities[defaulted_positions],
                    self.external_assets[defaulted_positions] + payments_received[defaulted_positions])

        elif calculation_method == 'Iterate':
            payments = self.total_liabilities.copy()
            loop = True
            iteration = 1

            while loop:
                new_payments = payments.copy()
                defaulted_positions = self.external_assets + np.matmul(self.payment_matrix.T, payments) < self.total_liabilities
                liquidated_value = np.maximum(0, self.external_assets + np.matmul(self.payment_matrix.T, payments))
                new_payments[defaulted_positions] = liquidated_value[defaulted_positions]
                loop = np.linalg.norm(new_payments - payments) > tolerance
                if iteration >= max_iterations:
                    print('No convergence!')
                    loop = False
                payments = new_payments
                iteration += 1

        else:
            raise ValueError('calcPayments: Unknown method - must be Standard or Iterate')

        return payments

    def get_defaulted_nodes_before_clearing(self):
        equity = self.external_assets + np.matmul(self.payment_matrix.T, self.total_liabilities) - self.total_liabilities
        return equity < 0

    def get_defaulted_nodes_after_clearing(self):
        payments = self.calc_payments()
        new_equity = self.external_assets + np.matmul(self.payment_matrix.T, payments) - self.total_liabilities
        return new_equity < 0

    def get_payment_matrix(self):
        payments = self.calc_payments()
        payments_made = self.payment_matrix * payments[:, None]
        return payments_made

    def get_liability_matrix_after_clearing(self):
        payments = self.calc_payments()
        liability_matrix_after_clearing = self.payment_matrix * payments[:, None]
        return liability_matrix_after_clearing

    def get_final_external_assets(self):
        payments = self.calc_payments()
        external_assets_after_clearing = self.external_assets + np.matmul(self.payment_matrix.T, payments) - payments
        return external_assets_after_clearing
