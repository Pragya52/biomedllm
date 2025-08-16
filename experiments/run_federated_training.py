import argparse
import logging
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
import os
import sys
import json
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.client_config import ClientConfig
from config.server_config import ServerConfig
from client.federated_client import FederatedQAClient
from server.federated_server import FederatedQAServer
from data.medical_qa_datasets import prepare_medical_qa_data
from experiments.evaluate_model import generate_experiment_summary



def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('federated_medical_qa.log'),
            logging.StreamHandler()
        ]
    )

def run_server(server_config: ServerConfig):
    """Run federated server."""
    server = FederatedQAServer(server_config)
    server.start_server()

def run_client(client_config: ClientConfig, num_rounds: int):
    """Run federated client."""
    
    # Wait for server to be ready
    time.sleep(10)
    
    try:
        client = FederatedQAClient(client_config)
        
        # Check server connectivity
        if not client.communicator.check_server_status():
            logging.error(f"Client {client_config.client_id} cannot connect to server")
            return
        
        # Start training
        logging.info(f"Starting training for {client_config.client_id}")
        training_history = client.train(num_rounds)
        
        # Save client results
        results_path = f"./results/client_{client_config.client_id}_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logging.info(f"Client {client_config.client_id} training completed")
        
    except Exception as e:
        logging.error(f"Error in client {client_config.client_id}: {e}")

def create_client_configs(args) -> List[ClientConfig]:
    """Create configurations for multiple clients."""
    client_configs = []
    
    for i in range(args.num_clients):
        config = ClientConfig(
            # Model configuration
            model_name=args.model_name,
            model_revision=args.model_revision,
            split_layer=args.split_layer,
            local_layers=args.local_layers,
            
            # QA configuration
            qa_format=args.qa_format,
            max_answer_length=args.max_answer_length,
            max_sequence_length=args.max_sequence_length,
            
            # Training configuration
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            
            # Privacy configuration
            gaussian_noise_std=args.noise_std,
            quantization_bits=args.quantization_bits,
            
            # Client specific
            client_id=f"client_{i+1}",
            server_url=f'http://localhost:{args.server_port}',
            device=args.device,
            mixed_precision=args.mixed_precision
        )
        
        client_configs.append(config)
    
    return client_configs

def main():
    """Main function to run federated learning with medical QA."""
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        
    parser = argparse.ArgumentParser(description='Federated Learning with BioMedLM for Medical QA')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='medqa', 
                       choices=['medqa', 'pubmedqa', 'bioasq', 'mmlu_medicine', 'medmcqa'],
                       help='Medical QA dataset to use')
    parser.add_argument('--split_strategy', type=str, default='specialty_based',
                       choices=['specialty_based', 'question_type', 'iid'],
                       help='Strategy for splitting data across clients')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='stanford-crfm/BioMedLM',
                       help='BioMedLM model name')
    parser.add_argument('--model_revision', type=str, default='main',
                       help='Model revision/branch')
    parser.add_argument('--split_layer', type=int, default=16,
                       help='Layer to split the model at')
    parser.add_argument('--local_layers', type=int, default=4,
                       help='Number of local processing layers')
    
    # QA task arguments
    parser.add_argument('--qa_format', type=str, default='multiple_choice',
                       choices=['multiple_choice', 'extractive', 'generative'],
                       help='QA task format')
    parser.add_argument('--max_answer_length', type=int, default=128,
                       help='Maximum answer length')
    parser.add_argument('--max_sequence_length', type=int, default=1024,
                       help='Maximum input sequence length')
    
    # Federated learning arguments
    parser.add_argument('--num_clients', type=int, default=3,
                       help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=20,
                       help='Number of federated learning rounds')
    parser.add_argument('--server_port', type=int, default=5000,
                       help='Server port')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per client')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--local_epochs', type=int, default=2,
                       help='Local epochs per round')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    
    # Privacy arguments
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='Gaussian noise standard deviation')
    parser.add_argument('--quantization_bits', type=int, default=8,
                       help='Quantization bits')
    
    # Technical arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Auto-detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Prepare medical QA data
    logging.info(f"Preparing {args.dataset} dataset for {args.num_clients} clients")
    prepare_medical_qa_data(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        split_strategy=args.split_strategy
    )
    
    # Server configuration
    server_config = ServerConfig(
        model_name=args.model_name,
        model_revision=args.model_revision,
        split_layer=args.split_layer,
        num_clients=args.num_clients,
        qa_format=args.qa_format,
        global_rounds=args.num_rounds,
        learning_rate=args.learning_rate,
        port=args.server_port,
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    # Create client configurations
    client_configs = create_client_configs(args)
    
    # Log experiment configuration
    logging.info("="*50)
    logging.info("FEDERATED MEDICAL QA EXPERIMENT")
    logging.info("="*50)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"QA Format: {args.qa_format}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Split Layer: {args.split_layer}")
    logging.info(f"Clients: {args.num_clients}")
    logging.info(f"Rounds: {args.num_rounds}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Mixed Precision: {args.mixed_precision}")
    logging.info("="*50)
    
    # Start server in a separate process
    logging.info("Starting federated server...")
    server_process = mp.Process(target=run_server, args=(server_config,))
    server_process.start()
    
    # Wait for server to start
    time.sleep(15)
    
    try:
        # Start clients in parallel
        logging.info("Starting federated clients...")
        with ThreadPoolExecutor(max_workers=args.num_clients) as executor:
            futures = []
            for client_config in client_configs:
                future = executor.submit(run_client, client_config, args.num_rounds)
                futures.append(future)
            
            # Wait for all clients to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Client execution error: {e}")
    
    finally:
        # Terminate server
        logging.info("Terminating server...")
        server_process.terminate()
        server_process.join()
    
    logging.info("Federated medical QA training completed!")
    
    # Generate summary
    try:
        from ..experiments.evaluate_model import generate_experiment_summary
        generate_experiment_summary(args.results_dir, args.num_clients)
    except Exception as e:
        logging.warning(f"Could not generate summary: {e}")

if __name__ == "__main__":
    main()
