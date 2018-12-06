package distributed_perceptron;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import peersim.config.Configuration;
import peersim.core.Cleanable;
import peersim.core.CommonState;
import peersim.core.Fallible;
import peersim.core.Network;
import peersim.core.Node;
import peersim.core.Protocol;
import weka.classifiers.functions.SGD;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;


public class perceptronNode implements Node {
	

	/**
	 * 
	 */
	
	// ================= fields ========================================
	// =================================================================

	/**
	 * New config option added to get the resourcepath where the resource file
	 * should be generated. Resource files are named <ID> in resourcepath.
	 * @config
	 */
	private static final String PAR_PATH = "resourcepath";

	/** used to generate unique IDs */
	private static long counterID = -1;

	/**
	 * The protocols on this node.
	 */
	protected Protocol[] protocol = null;

	/**
	 * The current index of this node in the node
	 * list of the {@link Network}. It can change any time.
	 * This is necessary to allow
	 * the implementation of efficient graph algorithms.
	 */
	private int index;

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	/**
	 * The ID of the node. It should be final, however it can't be final because
	 * clone must be able to set it.
	 */
	private long ID;

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	private String resourcepath;
	
	/** Normalize the training data */
    protected Normalize m_normalize;
    
	/**
	 * The training dataset
	 */
	public Instances traindataset;
	public Instances traindata;
	// The weight vector at the node
	public double[] wtvector;
	// The loss at the node
	public double[] loss;
	public int numFeat;
	
	public double bias;
	
	public perceptronNode(String prefix) 
	{
		String[] names = Configuration.getNames(PAR_PROT);
		resourcepath = (String)Configuration.getString(prefix + "." + PAR_PATH);
		System.out.println("Data is saved in: " + resourcepath);
		CommonState.setNode(this);
		ID = nextID();
		protocol = new Protocol[names.length];
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i);
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			protocol[i] = p; 
		}
	}

	private long nextID() {
		return counterID++;
	}

	public long getID() {
		return ID;
	}

	public int getIndex() {
		return index;
	}

	public Protocol getProtocol(int i) {
		return protocol[i];
	}
	
	public String getResourcePath()
	{
	  return resourcepath;
	}

	public int protocolSize() {
		return protocol.length;
	}

	public void setIndex(int index) {
		this.index = index;
		
	}

	public int getFailState() {return failstate;
	}

	public boolean isUp() {return failstate==OK; 
	}

	public void setFailState(int failState) {
		// after a node is dead, all operations on it are errors by definition
		if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
				"Cannot change fail state: node is already DEAD");
		switch(failState)
		{
		case OK:
			failstate=OK;
			break;
		case DEAD:
			//protocol = null;
			index = -1;
			failstate = DEAD;
			for(int i=0;i<protocol.length;++i)
				if(protocol[i] instanceof Cleanable)
					((Cleanable)protocol[i]).onKill();
			break;
		case DOWN:
			failstate = DOWN;
			break;
		default:
			throw new IllegalArgumentException(
					"failState="+failState);
		}
		
	}
	
	/** This function creates node with initial random weight vectors. **/
	public Object clone() 
	{
		perceptronNode result = null;
		try { result=(perceptronNode)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		result.setProtocol(new Protocol[protocol.length]);
		CommonState.setNode(result);
		result.setID(nextID());
		for(int i=0;i<getProtocol().length;i++)
		{
			CommonState.setPid(i);
			result.getProtocol()[i] = (Protocol)getProtocol()[i].clone();
		}
		
		/** Reading a chunk of arff file **/
		try{
		 String trainfilename = resourcepath + "/" + "eggEye_train" + result.getID() + ".arff";
	 
		 // set up the data 
	     FileReader reader = new FileReader(trainfilename);
	     Instances data = new Instances (reader);
	     
	     /** Convert to train files to csv for creating union data **/
	     String trainfilenameToWrite = resourcepath + "/" + "train_1_" + result.getID() + ".csv";
	 
		 /** set up the data **/
	     BufferedWriter outTrainFile = null;

	     	int[] removeArray=new int[1];
	     	removeArray[0] = data.numAttributes()-1;
			Remove removeFilter = new Remove();
			removeFilter.setAttributeIndicesArray(removeArray);
			removeFilter.setInvertSelection(false);
			removeFilter.setInputFormat(data);
			Instances newData = Filter.useFilter(data, removeFilter);
		 try{
				FileWriter fstreamTrain = new FileWriter(trainfilenameToWrite, true);
				outTrainFile = new BufferedWriter(fstreamTrain);
				
			    for(int bTrain=0;bTrain<data.numInstances();bTrain++)
			    {
			    	outTrainFile.write(String.valueOf(newData.instance(bTrain)));
			    	outTrainFile.write("\n");
			    }
			    outTrainFile.close();
		     }
		 catch (IOException ioe) 
			{ioe.printStackTrace();}
	     
	     /** Make the last attribute be the class **/
	     int classIndex = data.numAttributes()-1;
	     data.setClassIndex(classIndex); 
	    
	     /**Get the number of instances and attributes **/
	     int numInst = data.numInstances();
	     numFeat=data.numAttributes()-1;
	     String testfilename = resourcepath + "/" + "eggEye_test" + result.getID() + ".arff";
	     String testfilenameToWrite = resourcepath + "/" + "test" + result.getID() + ".csv";
	     FileReader readerTest = new FileReader(testfilename);
	     Instances testdata = new Instances (readerTest);
	     BufferedWriter outTestFile = null;
		 try{
				FileWriter fstreamTest = new FileWriter(testfilenameToWrite, true);
				outTestFile = new BufferedWriter(fstreamTest);
				
			    for(int b=0;b<testdata.numInstances();b++)
			    {
			    	outTestFile.write(String.valueOf(testdata.instance(b)));
			    	outTestFile.write("\n");
			    }
			    outTestFile.close();
		     }
		 catch (IOException ioe) 
			{ioe.printStackTrace();}
	     
	     result.traindataset=data;
	     result.traindata=newData;
	    
	     /** Initialize the weight vector with random values **/ 
	     double[] wtVec = new double[numFeat];
	     for(int k=0; k<numFeat; k++)
	     { wtVec[k]=(Math.random() * ((0.01 - (-0.01)) )) -0.01;
	    // System.out.print(wtVec[k]+"\n ");
	     }

	     perceptronVector vWt = new perceptronVector(wtVec);
	     SGD sgd = new SGD();
	     //sgd.buildClassifier(data);
	     bias=(Math.random() * ((0.01 - (-0.01)) )) -0.01;
	     double[] wx = new double[numInst];
	     double y;
	     double [] cur_loss = new double[numInst];
	     //System.out.print(wtVec+"\n ");
	     
	     /** Iterating through the rows **/
	     for(int m=0;m<numInst;m++)
	     { 
	      
	      /** Dot product of input instance and weight vector. **/
	      wx[m] = bias + sgd.dotProd(newData.instance(m), wtVec,1);
	     // System.out.print(wx[m]+"\n ");
	      
	      /** Getting sign of the output **/
	      y=Math.signum(wx[m]);
//	      /** Acts as a sign function **/
//	      if(wx[m]>0) {
//	    	  y=1;
//	      }
//	      else {
//	    	  y=-1;
//	      }
	      
	      /** Error **/
	      cur_loss[m]= data.instance(m).classValue()-y;

	     }
	     result.wtvector=wtVec;
	     result.loss = cur_loss;	     
	     
	    
	     }
		 catch(Exception e)
		 {e.printStackTrace(); }
	        
		System.out.println("\n"+"created node with ID: " + result.getID());
		//System.out.println("\n"+"Loss: " + result.loss);
		return result;
		
	}
	
	private Protocol[] getProtocol() {
		return protocol;
	}

	private void setID(long iD) {
		ID = iD;
	}

	private void setProtocol(Protocol[] protocols) {
		this.protocol = protocol;
	}

	public String toString() 
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("ID: "+ID+" index: "+index+"\n");
		for(int i=0; i<protocol.length; ++i)
		{
			buffer.append("protocol[" + i +"]=" + protocol[i] + "\n");
		}
		return buffer.toString();
	}
	
	/** Implemented as <code>(int)getID()</code>. */
	public int hashCode() { return (int)getID(); }
	
	/**
	 * This method is called at the end of simulation cycles. And it writes final 
	 * global weights obtained to global_<id>.dat files in resourcepath
	 */
	public void writeLossNorm(double ls) 
	{
		String filename = resourcepath + "/" + "loss_norm_base_" + this.getID() + ".txt";
		//String filename = resourcepath + "/" + "wt_coeff_" + this.getID() + ".txt";
		BufferedWriter out = null;
		try {
			FileWriter fstream = new FileWriter(filename, true);
			out = new BufferedWriter(fstream);
			//IlpVector vLoss = new IlpVector(ls);
		    out.write(String.valueOf(ls)); 
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
	
	/** Function which Writes csv file **/
	public void write(String fName, double ls) 
	{
		BufferedWriter out = null;
		try {
			FileWriter fstream = new FileWriter(fName, true);
			out = new BufferedWriter(fstream);
			//IlpVector vLoss = new IlpVector(ls);
		    out.write(String.valueOf(ls)); 
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
