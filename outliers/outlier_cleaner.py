#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    top_residuals = sorted([p-n for (p,n) in zip(predictions,net_worths)],reverse=True)[:9]
    for (p,a,n) in zip(predictions,ages,net_worths):
        residual = p-n
        if residual in top_residuals:
            pass
        else:
            cleaned_data.append((a,n,residual))
    return cleaned_data
