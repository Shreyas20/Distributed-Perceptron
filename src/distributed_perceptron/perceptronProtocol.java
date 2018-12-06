package distributed_perceptron;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
//import java.util.List;
import java.util.Random;

import peersim.cdsim.CDProtocol;
import peersim.cdsim.CDState;
import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.CommonState;
import peersim.core.Linkable;
import peersim.core.Node;
import weka.classifiers.functions.SGD;

public class perceptronProtocol implements CDProtocol {
	/**
	 * New config option to get the learning parameter lambda
	 * @config
	 */
	private static final String PAR_LR = "lr";
	/**
	 * New config option to get the learning parameter lambda
	 * @config
	 */
	//private static final String PAR_ALPHA = "alpha";
	/**
	 * New config option to get the number of iteration
	 * @config
	 */
	//New config option to get the user-defined threshold epsilon
	private static final String PAR_EPSILON = "epsilon";
	
	/** Learning parameter */
	protected double lr;
	/** Learning rate */
	//protected double alpha;
	/** Number of iteration (T in algorithm)*/
	protected int iter;
	/** Linkable identifier */
	protected int lid;
	/** 2d coordinates components. */
    //private double x, y;
    /** User defined threshold */
    private double epsilon;
	
	public perceptronProtocol(String prefix) {
		lr = Configuration.getDouble(prefix + "." + PAR_LR);
		//iter = Configuration.getInt(prefix + "." + PAR_ITERATION);
		epsilon=Configuration.getDouble(prefix + "."+ PAR_EPSILON);
		lid = FastConfig.getLinkable(CommonState.getPid());
		//x = y = -1;
	}
	/**
	 * Clone an existing instance. The clone is considered 
	 * new
	 */
	public Object clone() {
		perceptronProtocol perceptron = null;
		try { perceptron = (perceptronProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return perceptron;
	}
	public void nextCycle(Node node, int pid) 
	{
		/** Get the cycle number **/
		int iterNum = CDState.getCycle()+1;
		//System.out.println("Cycle number is "+CDState.getCycle());
		/** Record start time of cycle **/
		double startTime=System.currentTimeMillis();
		
		/** Obtain the weight and local loss for the node **/
		perceptronNode pn = (perceptronNode)node;
		double[] wts = new double[pn.traindataset.numAttributes()-1];
		double[] lastWts=new double[pn.traindataset.numAttributes()];
		//double[] NmLoss=new double[pn.traindataset.numInstances()];
		//boolean[] lessEps = new boolean[pn.traindataset.numInstances()];
		boolean lessEps;
		double[] local_loss = new double[pn.traindataset.numInstances()];
		boolean finalBool=true;
		//double totalNm=0.0;		
		double bias;
		wts = pn.wtvector;
		local_loss=pn.loss;
		bias=pn.bias;
		
		//Write the initial weights to the file
		if(iterNum==1)
		{
		/** Print these initial weights to a file **/
		String fn = pn.getResourcePath() + "/" + "wtVec" + pn.getID() + ".txt";
		writeLoss(fn,pn.bias);
		for(int f=0;f<pn.traindataset.numAttributes()-1;f++)
		{
			//pn.wtvector[f]=pn.wtvector[f]/vec.magnitude();	
			//System.out.print(pn.loss[f]+ " ");
			writeLoss(fn,pn.wtvector[f]);
		}
		}
    	if(iterNum>1)
		{
		/** Should this cycle be executed?
		Check if the epsilon tolerance on weight vectors
	     is less than the user-defined threshold. If so, stop.
		Read in the attribute file
		Get the number of attributes in this file **/
		int numAttrb=pn.traindataset.numAttributes()-1;
		int numInst=pn.traindataset.numInstances();
		String path = pn.getResourcePath() + "/" + "wtVec" + pn.getID() + ".txt";
		FileReader reader; 
		String currLine;
		int currLineNo;
		int counter = 0;
		
		/** This try catch block gets number of lines in the weight file **/
		try 
		{
			reader = new FileReader(path);
			LineNumberReader lnr = new LineNumberReader(reader);
//			// read lines till the end of the stream
	         while((currLine=lnr.readLine())!=null)
	         {
	            counter++;
	         }
		} catch (Exception e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		/** This block gets last weight vectors by storing last no of attributes times values in a double arraylist **/ 
		try 
		{
			reader = new FileReader(path);
			LineNumberReader lnr = new LineNumberReader(reader);
			int d=0;
	         while((currLine=lnr.readLine())!=null)
	         {
	        	 currLineNo=lnr.getLineNumber();
	        	if(currLineNo > counter - pn.traindataset.numAttributes()) {
	            	lastWts[d]=Double.parseDouble(currLine);
	            	d=d+1;
	        	}
	            }
	         }
		 catch (Exception e)
		{
//			// TODO Auto-generated catch block
			e.printStackTrace();
		}
//**********		
		/** Compute the difference in 2-norms of current and last weight vectors **/
		double loss=0;
		 lessEps= false;
		 
		 /** calculating 2-norms of current weights **/
		double Cnorm=pn.bias*pn.bias;
		
		for(int c=0;c<pn.traindataset.numAttributes()-1;c++) {
			Cnorm=Cnorm+(wts[c]*wts[c]);
		}
		double c2norm=Math.sqrt(Cnorm);
		
		 /** calculating 2-norms of last weights **/
		double onorm=0;
		for(int o=0;o<pn.traindataset.numAttributes()-1;o++) {
			onorm=onorm+(lastWts[o]*lastWts[o]);
		}
		double o2norm=Math.sqrt(onorm);
		

		if(Math.abs((c2norm-o2norm)/numAttrb)<epsilon) {
			lessEps=true;
		}
		 finalBool=finalBool && lessEps;
		
		System.out.println("Avg Norm of consecutive feature weights is "+Math.abs((c2norm-o2norm)/numAttrb));
		
		/** If the user-defined threshold is reach, exit from simulation. **/
		if(finalBool) {
			System.out.println("\n\n");
			System.out.println("****TERMINATED****");
			System.out.println("\n\n");
		 pn.setFailState(2);
		}
		
		} // end if iterNum >1 
//**************		
		
		/** Gossip with a neighbor **/
		perceptronNode peer = (perceptronNode)selectNeighbor(node, pid);
		System.out.println("Node [" + pn.getID() + "] is gossiping with Node [" + peer.getID() + "]" );
		//System.out.print(wts+"\n ");
		double[] peerWt =  new double[pn.traindataset.numAttributes()-1];
		peerWt=peer.wtvector;
		double peerBias=peer.bias;
		
		/** Add local weights with peer's weights  **/
		double[] updateWt = new double[pn.traindataset.numAttributes()-1];
		// The loss needs to be in a loop 
		for (int j=0;j<updateWt.length;j++)
		{updateWt[j]=(wts[j]+peerWt[j])/2;
		 //System.out.print(updateLoss[j]+" ");
		}
		double updatedBias=(bias+peerBias)/2;

		pn.wtvector=updateWt;
		//Set peer's loss vector also
		peer.wtvector = updateWt;
		
		pn.bias=updatedBias;
		peer.bias=updatedBias;
		
		/** Do the stochastic gradient descent step 
		 	Randomly pick a sample from the dataset **/
//		Random rn = new Random();
//		int rnd = rn.nextInt(pn.traindataset.numInstances()-1);
		//System.out.println("Choosing instance no "+rnd+ " in node "+ pn.getID());
		//SGD anotherSGD = new SGD();
		/** Estimate the gradient **/
		
		double [] cur_loss = new double[pn.traindataset.numInstances()];
		SGD sgd = new SGD();
		double newY;
//		for(int a=0; a< pn.traindata.numAttributes();a++) {
//		System.out.print(pn.wtvector[a]+" ");
//		}
		double [] newWx = new double[pn.traindataset.numInstances()];
	    for(int p=0;p<pn.traindataset.numInstances();p++)
	     { 
	    	
	    	/** Getting dot product of data instance and weight vector **/
	      newWx[p] = pn.bias+sgd.dotProd(pn.traindata.instance(p), pn.wtvector, 1);
	      
	      /** Getting sign of the dot product **/
	      newY=Math.signum(newWx[p]);
//	      if(newWx[p]>0) {
//	    	  newY=1;
//	      }
//	      else {
//	    	  newY=-1;
//	      }
	      
	      /** calculating error **/
	      cur_loss[p]= pn.traindataset.instance(p).classValue()-newY;
	      
	      
	      /** Updating bias value **/
    	  pn.bias=pn.bias+2*lr*cur_loss[p];
    	  
    	  /** Updating all other weight values **/
	      for(int i=0; i< pn.traindataset.numAttributes()-1; i++) {
	    	  pn.wtvector[i]=pn.wtvector[i]+2*lr*cur_loss[p]*pn.traindataset.instance(p).value(i)*pn.wtvector[i];
	      }
	     }	
	    
	   // perceptronVector vLoss = new perceptronVector(cur_loss);
	   
	    pn.loss=cur_loss;
	    
	    /** Writing weight vectors in the respective files created for each node **/

	    String filename = pn.getResourcePath() + "/" + "wtVec" + pn.getID() + ".txt";
		writeLoss(filename,pn.bias);
		for(int f=0;f<pn.traindataset.numAttributes()-1;f++)
		{
			writeLoss(filename,pn.wtvector[f]);
		}
		
		/** Writing time required for computation for the node in the same **/
	    String timeFile = pn.getResourcePath() + "/" + "time_Vec.txt";
	    double timeDist = System.currentTimeMillis() - startTime;
	    
	    writeLoss(timeFile,timeDist);
	    
	    /** Writing training errors in the respective files created for each node **/

	    String lossFile = pn.getResourcePath() + "/" + "loss" +pn.getID()+ ".txt";
	    double SE=0;
	    for(int l=0;l<pn.traindataset.numInstances();l++) {
	    	SE=SE+(cur_loss[l]*cur_loss[l]);
	 
	    }
	    float RMSE=(float) (Math.sqrt(SE)/(pn.traindataset.numInstances()));
	    
	    writeLoss(lossFile,RMSE);
	    //System.out.println("Root mean square error for "+CDState.getCycle()+"th cycle and node ID "+pn.getID()+" is "+RMSE);
	    
		//
		} // end of method

	/** Function to randomly select a neighbor for gossiping **/
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}
	
	public void writeLoss(String fName, double db)
	{
		BufferedWriter out = null;
		try
		{
			FileWriter fstream = new FileWriter(fName, true);
			out = new BufferedWriter(fstream);
			out.write(String.valueOf(db)); 
		    out.write("\n");
	    	}
		catch (IOException ioe) 
		{ioe.printStackTrace();}
		finally
		{
		if (out != null) 
	    {try {out.close();} catch (IOException e) {e.printStackTrace();}
	    }
		}
	}
}