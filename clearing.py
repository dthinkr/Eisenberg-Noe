""" second round clearing by Eisenburg Noe
"""
import numpy as np

class ClearingSystem:
    def __init__(self, vecE, matL):
        self.vecE = vecE
        self.matL = matL
        self.numBanks = len(vecE)
        self.vecPbar = np.sum(matL, axis=1, dtype=float)
        self.matPi = matL / self.vecPbar[:, None]
        self.matPi[np.isnan(self.matPi)] = 0

    def calcPayments(self, strCalculationMethod="Standard", numMaxIterations=500, dblTolerance=0.0001):
        if strCalculationMethod == 'Standard':
            vecPayments = self.vecPbar.copy()
            vecPaymentsPrev = np.zeros_like(vecPayments)

            while np.any(vecPayments != vecPaymentsPrev):
                vecPaymentsPrev = vecPayments.copy()
                vecPaymentsReceived = np.matmul(self.matPi.T, vecPayments)
                posDefaulted = self.vecE + vecPaymentsReceived < self.vecPbar
                vecPayments[posDefaulted] = np.minimum(self.vecPbar[posDefaulted], self.vecE[posDefaulted] + vecPaymentsReceived[posDefaulted])

        elif strCalculationMethod == 'Iterate':
            vecPayments = self.vecPbar.copy()
            blnLoop = True
            iIteration = 1

            while blnLoop:
                vecPnew = vecPayments.copy()
                posDefaulted = self.vecE + np.matmul(self.matPi.T, vecPayments) < self.vecPbar
                vecLiquidatedValue = np.maximum(0, self.vecE + np.matmul(self.matPi.T, vecPayments))
                vecPnew[posDefaulted] = vecLiquidatedValue[posDefaulted]
                blnLoop = np.linalg.norm(vecPnew - vecPayments) > dblTolerance
                if iIteration >= numMaxIterations:
                    print('No convergence!')
                    blnLoop = False
                vecPayments = vecPnew
                iIteration += 1

        else:
            raise ValueError('calcPayments: Unknown method - must be Standard or Iterate')

        return vecPayments

    def getDefaultedNodesBeforeClearing(self):
        vecEquity = self.vecE + np.matmul(self.matPi.T, self.vecPbar) - self.vecPbar
        return vecEquity < 0

    def getDefaultedNodesAfterClearing(self):
        vecPayments = self.calcPayments()
        vecEquityNew = self.vecE + np.matmul(self.matPi.T, vecPayments) - self.vecPbar
        return vecEquityNew < 0

    def getPaymentMatrix(self):
        vecPayments = self.calcPayments()
        matPaymentsMade = self.matPi * vecPayments[:, None]
        return matPaymentsMade

    def getLiabilityMatrixAfterClearing(self):
        vecPayments = self.calcPayments()
        matL_after_clearing = self.matPi * vecPayments[:, None]
        return matL_after_clearing

    def getFinalExternalAssets(self):
        vecPayments = self.calcPayments()
        vecE_after_clearing = self.vecE + np.matmul(self.matPi.T, vecPayments) - vecPayments
        return vecE_after_clearing