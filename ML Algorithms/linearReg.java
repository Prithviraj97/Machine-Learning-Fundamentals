public class linearReg{
    public static void main(String[] args){
        double[] x = {1,2,3,4,5};
        double[] y = {2,3,4,5,6};
        double[] theta = {0,0};
        double alpha = 0.01;
        int m = x.length;
        int n = theta.length;
        int iterations = 1000;
        for(int i=0;i<iterations;i++){
            double[] h = new double[m];
            for(int j=0;j<m;j++){
                h[j] = 0;
                for(int k=0;k<n;k++){
                    h[j] += theta[k]*x[j];
                }
            }
            double[] temp = new double[n];
            for(int j=0;j<n;j++){
                temp[j] = 0;
                for(int k=0;k<m;k++){
                    temp[j] += (h[k]-y[k])*x[k];
                }
                temp[j] /= m;
            }
            for(int j=0;j<n;j++){
                theta[j] -= alpha*temp[j];
            }
        }
        System.out.println("Theta values are: ");
        for(int i=0;i<n;i++){
            System.out.println(theta[i]);
        }
    }
}