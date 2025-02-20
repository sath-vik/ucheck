# epoch,lr,loss,iou,val_loss,val_iou,val_dice
# 0,9.779754323328192e-05,1.7954946637634954,0.41433551994780354,1.236297409057617,0.532657131910324,0.6875998845100403
# 1,9.140576474687264e-05,1.165393681336861,0.4880867937772784,0.890328004360199,0.5704933869838714,0.7191721034049988
# 2,8.14503363531613e-05,0.9521872348291235,0.524308265679457,0.8249578275680542,0.6045948958396912,0.7457278685569764
# 3,6.890576474687264e-05,0.8715361343097558,0.5474366478291045,0.7788577547073364,0.5975239515304566,0.7405792956352234
# 4,5.500000000000001e-05,0.7930787099001385,0.5732943325793599,0.7013292720317841,0.6273262434005737,0.7638058166503906
# 5,4.109423525312737e-05,0.7585003369589353,0.5860920447359816,0.6737409813404083,0.6385322136878967,0.772345202922821
# 6,2.8549663646838717e-05,0.7251949387716573,0.6001279735549141,0.6365154800415039,0.655905902147293,0.7855544905662537
# 7,1.8594235253127375e-05,0.6846770599263834,0.6149955158037902,0.6238291757106781,0.6629532752037048,0.7906809253692627
# 8,1.2202456766718093e-05,0.6682307241662514,0.6223135055876355,0.615593106508255,0.6661333270072937,0.7929223313331604
# 9,1e-05,0.6568639475756988,0.6271622135212411,0.601269129037857,0.6706046624183655,0.7965268783569336



import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path, output_dir="plots"):
    # Load the CSV file
    log = pd.read_csv(csv_path)
    
    # Ensure the output directory exists
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(log['epoch'], log['loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(log['epoch'], log['val_loss'], label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.show()

    # Plot Training and Validation IoU
    plt.figure(figsize=(10, 6))
    plt.plot(log['epoch'], log['iou'], label='Train IoU', color='green', marker='o')
    plt.plot(log['epoch'], log['val_iou'], label='Validation IoU', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/iou_plot.png")
    plt.show()

    # Plot Validation Dice Score
    if 'val_dice' in log.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(log['epoch'], log['val_dice'], label='Validation Dice', color='purple', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/dice_plot.png")
        plt.show()

    # Plot Learning Rate
    if 'lr' in log.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(log['epoch'], log['lr'], label='Learning Rate', color='brown', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/lr_plot.png")
        plt.show()

if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "PreviousRuns/Model3/100_epochs_log.csv"  # Replace with the actual path to your log.csv file
    
    # Call the function to plot metrics
    plot_metrics(csv_file)
