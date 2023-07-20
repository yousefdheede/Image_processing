import java.util.Scanner;
public class classa {
public static int sol(int T,int N){
    int s=0;
   for(int i=0;i<T;i++){
    if(N>=1&&N<=1000000000){
      while(N>=5){
          s+=1;
          N=N-5;
      }
      System.out.print(s);

   }
}
   return s;
}
    
    
 public static void main( String args[] ) {
    try (Scanner inn = new Scanner( System.in )) {
        int t,n;
        t=inn.nextInt();
        n=inn.nextInt();
        sol(t,n);
    }
    
    }
}
