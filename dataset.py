# dataset.py - Phiên bản tương thích ngược với tối ưu memory và cache

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc
import os
import pickle
import hashlib
import time
from pathlib import Path
import json


class CacheManager:
    """
    Quản lý cache cho dataset - tự động detect changes và invalidate cache
    """
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
    def _get_file_hash(self, file_path, sample_size=10000):
        """
        Tạo hash nhanh của file bằng cách sample một số dòng đầu và cuối
        """
        hasher = hashlib.md5()
        
        # File stats
        stat = os.stat(file_path)
        hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())
        
        # Sample content from beginning and end
        with open(file_path, 'rb') as f:
            # First chunk
            chunk = f.read(1024 * 1024)  # 1MB
            hasher.update(chunk)
            
            # Last chunk if file is large enough
            if stat.st_size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)  # Go to 1MB before end
                chunk = f.read()
                hasher.update(chunk)
                
        return hasher.hexdigest()
    
    def _get_config_hash(self):
        """
        Hash của config parameters ảnh hưởng đến data processing
        """
        config_params = {
            'MAX_SEQ': getattr(Config, 'MAX_SEQ', 100),
            'BATCH_SIZE': getattr(Config, 'BATCH_SIZE', 32),
        }
        return hashlib.md5(json.dumps(config_params, sort_keys=True).encode()).hexdigest()
    
    def get_cache_metadata(self):
        """
        Đọc metadata của cache
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache_metadata(self, data):
        """
        Lưu metadata của cache
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_cache_valid(self, cache_key, source_file):
        """
        Kiểm tra cache có còn valid không
        """
        metadata = self.get_cache_metadata()
        
        if cache_key not in metadata:
            return False
            
        cache_info = metadata[cache_key]
        
        # Check file hash
        current_hash = self._get_file_hash(source_file)
        if cache_info.get('file_hash') != current_hash:
            return False
            
        # Check config hash
        current_config_hash = self._get_config_hash()
        if cache_info.get('config_hash') != current_config_hash:
            return False
            
        # Check if cache files exist
        for file_path in cache_info.get('cache_files', []):
            if not os.path.exists(file_path):
                return False
                
        return True
    
    def register_cache(self, cache_key, source_file, cache_files):
        """
        Đăng ký cache mới
        """
        metadata = self.get_cache_metadata()
        
        metadata[cache_key] = {
            'file_hash': self._get_file_hash(source_file),
            'config_hash': self._get_config_hash(),
            'cache_files': cache_files,
            'created_at': time.time(),
            'source_file': str(source_file)
        }
        
        self.save_cache_metadata(metadata)
    
    def get_cache_path(self, cache_key, suffix=""):
        """
        Tạo đường dẫn cache file
        """
        return self.cache_dir / f"{cache_key}{suffix}"
    
    def clear_cache(self, cache_key=None):
        """
        Xóa cache (tất cả hoặc một cache cụ thể)
        """
        metadata = self.get_cache_metadata()
        
        if cache_key is None:
            # Clear all cache
            for key, info in metadata.items():
                for file_path in info.get('cache_files', []):
                    if os.path.exists(file_path):
                        os.remove(file_path)
            metadata.clear()
        else:
            # Clear specific cache
            if cache_key in metadata:
                info = metadata[cache_key]
                for file_path in info.get('cache_files', []):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                del metadata[cache_key]
        
        self.save_cache_metadata(metadata)
        print(f"Cache cleared: {cache_key if cache_key else 'all'}")


class DKTDataset(Dataset):
    """
    Dataset gốc - giữ nguyên interface để tương thích
    """
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        for id in self.samples.index:
            exe_ids, answers, ela_time, categories = self.samples[id]
            if len(exe_ids) > max_seq:
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append(
                        (exe_ids[l:l+max_seq], answers[l:l+max_seq], ela_time[l:l+max_seq], categories[l:l+max_seq]))
            elif len(exe_ids) < self.max_seq and len(exe_ids) > 50:
                self.data.append((exe_ids, answers, ela_time, categories))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, ela_time, exe_category = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros(self.max_seq, dtype=int)
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            elapsed_time[-seq_len:] = ela_time
            exe_cat[-seq_len:] = exe_category
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsed_time, 0, 0)
        input_rtime = np.delete(input_rtime, -1)

        input = {"input_ids": exe_ids, "input_rtime": input_rtime.astype(
            np.int64), "input_cat": exe_cat}
        return input, ans


def get_dataloaders_chunked(chunk_size=5000000, use_sample=False, sample_size=10000000, 
                          use_cache=True, cache_dir="./cache", force_refresh=False):
    """
    Phiên bản tối ưu memory của get_dataloaders() - TƯƠNG THÍCH hoàn toàn với cache
    
    Args:
        chunk_size: Kích thước chunk để đọc CSV
        use_sample: Có lấy mẫu hay không (để test nhanh)
        sample_size: Kích thước mẫu nếu use_sample=True
        use_cache: Có sử dụng cache không
        cache_dir: Thư mục lưu cache
        force_refresh: Bắt buộc refresh cache
    """
    
    # Initialize cache manager
    cache_manager = CacheManager(cache_dir) if use_cache else None
    
    # Create cache key based on parameters
    cache_key_params = {
        'chunk_size': chunk_size,
        'use_sample': use_sample,
        'sample_size': sample_size if use_sample else 'full',
        'max_seq': Config.MAX_SEQ
    }
    cache_key = f"grouped_data_{hashlib.md5(str(cache_key_params).encode()).hexdigest()}"
    
    # Check cache validity
    if use_cache and not force_refresh and cache_manager.is_cache_valid(cache_key, Config.TRAIN_FILE):
        print("Loading from cache...")
        try:
            # Load cached grouped data
            cache_path = cache_manager.get_cache_path(cache_key, "_grouped.pkl")
            with open(cache_path, 'rb') as f:
                train, val = pickle.load(f)
            
            print(f"Loaded from cache - Train size: {train.shape}, Validation size: {val.shape}")
            
            # Create datasets and dataloaders
            return _create_dataloaders_from_splits(train, val)
            
        except Exception as e:
            print(f"Cache loading failed: {e}. Falling back to normal processing...")
    
    # Normal processing path
    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'int8', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    
    print("Loading CSV with chunked processing...")
    
    # Đọc file theo chunks và xử lý từng phần
    chunk_reader = pd.read_csv(
        Config.TRAIN_FILE, 
        usecols=[1, 2, 3, 4, 5, 7, 8], 
        dtype=dtypes, 
        chunksize=chunk_size
    )
    
    processed_chunks = []
    total_rows = 0
    
    for chunk_idx, chunk in enumerate(chunk_reader):
        print(f"Processing chunk {chunk_idx + 1}, size: {len(chunk)}")
        
        # Áp dụng các filter giống như code gốc
        chunk = chunk[chunk.content_type_id == 0]
        chunk.prior_question_elapsed_time.fillna(0, inplace=True)
        chunk.prior_question_elapsed_time /= 1000
        chunk.prior_question_elapsed_time = chunk.prior_question_elapsed_time.astype(int)
        chunk = chunk.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
        
        processed_chunks.append(chunk)
        total_rows += len(chunk)
        
        # Nếu dùng sample mode và đã đủ data
        if use_sample and total_rows >= sample_size:
            print(f"Reached sample size limit: {sample_size}")
            break
        
        # Cleanup memory
        gc.collect()
    
    print(f"Total processed rows: {total_rows}")
    
    # Concatenate tất cả chunks
    print("Concatenating chunks...")
    train_df = pd.concat(processed_chunks, ignore_index=True)
    del processed_chunks
    gc.collect()
    
    # Nếu dùng sample, chỉ lấy phần đầu
    if use_sample and len(train_df) > sample_size:
        train_df = train_df.head(sample_size)
    
    print("Shape of dataframe:", train_df.shape)
    n_skills = train_df.content_id.nunique()
    print("No. of skills:", n_skills)
    
    # Grouping - giống như code gốc
    print("Grouping users...")
    group = train_df[["user_id", "content_id", "answered_correctly", "prior_question_elapsed_time", "task_container_id"]]\
        .groupby("user_id")\
        .apply(lambda r: (r.content_id.values, r.answered_correctly.values,
                          r.prior_question_elapsed_time.values, r.task_container_id.values))
    
    del train_df
    gc.collect()
    
    print("Splitting...")
    train, val = train_test_split(group, test_size=0.2, random_state=42)
    print("Train size:", train.shape, "Validation size:", val.shape)
    
    # Save to cache
    if use_cache:
        try:
            print("Saving to cache...")
            cache_path = cache_manager.get_cache_path(cache_key, "_grouped.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump((train, val), f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Register cache
            cache_manager.register_cache(cache_key, Config.TRAIN_FILE, [str(cache_path)])
            print(f"Cache saved to: {cache_path}")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    return _create_dataloaders_from_splits(train, val)


def _create_dataloaders_from_splits(train, val):
    """
    Helper function để tạo dataloaders từ train/val splits
    """
    # Tạo datasets - giống như code gốc
    train_dataset = DKTDataset(train, max_seq=Config.MAX_SEQ)
    val_dataset = DKTDataset(val, max_seq=Config.MAX_SEQ)
    
    # Tạo dataloaders với tối ưu memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=min(4, getattr(Config, 'MAX_WORKERS', 4)),
        shuffle=True,
        pin_memory=getattr(Config, 'PIN_MEMORY', False),
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=min(4, getattr(Config, 'MAX_WORKERS', 4)),
        shuffle=False,
        pin_memory=getattr(Config, 'PIN_MEMORY', False),
        persistent_workers=True
    )
    
    del train_dataset, val_dataset
    gc.collect()
    
    return train_loader, val_loader


def get_dataloaders(use_cache=True, cache_dir="./cache", force_refresh=False):
    """
    Hàm gốc - giữ nguyên tên để tương thích
    Tự động chuyển sang phiên bản tối ưu memory với cache
    """
    print("Using memory-optimized data loading with cache...")
    
    # Tự động điều chỉnh chunk_size dựa trên memory
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 8:
            chunk_size = 1000000  # 1M rows per chunk
            print(f"Low memory detected ({available_memory_gb:.1f}GB), using small chunks")
        elif available_memory_gb < 16:
            chunk_size = 3000000  # 3M rows per chunk  
            print(f"Medium memory detected ({available_memory_gb:.1f}GB), using medium chunks")
        else:
            chunk_size = 5000000  # 5M rows per chunk
            print(f"High memory detected ({available_memory_gb:.1f}GB), using large chunks")
            
    except ImportError:
        chunk_size = 2000000  # Default fallback
        print("Cannot detect memory, using default chunk size")
    
    return get_dataloaders_chunked(
        chunk_size=chunk_size, 
        use_cache=use_cache, 
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )


def get_dataloaders_sample(sample_size=5000000, use_cache=True, cache_dir="./cache", force_refresh=False):
    """
    Phiên bản để test nhanh với sample nhỏ - có cache
    """
    print(f"Loading sample dataset ({sample_size} rows) with cache...")
    return get_dataloaders_chunked(
        chunk_size=1000000, 
        use_sample=True, 
        sample_size=sample_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )


class ProgressiveDataLoader:
    """
    Loader dữ liệu từng phần cho dataset cực lớn - với cache support
    """
    
    def __init__(self, file_path, max_seq, batch_size, chunk_size=2000000, use_cache=True, cache_dir="./cache"):
        self.file_path = file_path
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.cache_manager = CacheManager(cache_dir) if use_cache else None
        
        self.dtypes = {
            'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
            'answered_correctly': 'int8', "content_type_id": "int8",
            "prior_question_elapsed_time": "float32", "task_container_id": "int16"
        }
        self.usecols = [1, 2, 3, 4, 5, 7, 8]
    
    def get_progressive_loaders(self, train_ratio=0.8, force_refresh=False):
        """
        Generator trả về train/val loaders cho từng chunk - với cache
        """
        cache_key_base = f"progressive_{self.chunk_size}_{train_ratio}_{self.max_seq}"
        
        chunk_reader = pd.read_csv(
            self.file_path,
            usecols=self.usecols,
            dtype=self.dtypes,
            chunksize=self.chunk_size
        )
        
        for chunk_idx, chunk in enumerate(chunk_reader):
            chunk_cache_key = f"{cache_key_base}_chunk_{chunk_idx}"
            
            # Try to load from cache first
            if (self.use_cache and not force_refresh and 
                self.cache_manager.is_cache_valid(chunk_cache_key, self.file_path)):
                
                try:
                    print(f"Loading chunk {chunk_idx + 1} from cache...")
                    cache_path = self.cache_manager.get_cache_path(chunk_cache_key, "_splits.pkl")
                    with open(cache_path, 'rb') as f:
                        train, val = pickle.load(f)
                    
                    # Create datasets and loaders
                    train_dataset = DKTDataset(train, max_seq=self.max_seq)
                    val_dataset = DKTDataset(val, max_seq=self.max_seq)
                    
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    
                    yield train_loader, val_loader, chunk_idx
                    
                    # Cleanup
                    del train, val, train_dataset, val_dataset, train_loader, val_loader
                    gc.collect()
                    continue
                    
                except Exception as e:
                    print(f"Failed to load chunk {chunk_idx + 1} from cache: {e}")
            
            # Normal processing
            print(f"Processing chunk {chunk_idx + 1}")
            
            # Preprocess chunk
            chunk = chunk[chunk.content_type_id == 0]
            if chunk.empty:
                continue
                
            chunk.prior_question_elapsed_time.fillna(0, inplace=True)
            chunk.prior_question_elapsed_time /= 1000
            chunk.prior_question_elapsed_time = chunk.prior_question_elapsed_time.astype(int)
            chunk = chunk.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
            
            # Group by user
            group = chunk[["user_id", "content_id", "answered_correctly", "prior_question_elapsed_time", "task_container_id"]]\
                .groupby("user_id")\
                .apply(lambda r: (r.content_id.values, r.answered_correctly.values,
                                  r.prior_question_elapsed_time.values, r.task_container_id.values))
            
            if len(group) == 0:
                continue
                
            # Split train/val
            train, val = train_test_split(group, test_size=1-train_ratio, random_state=42)
            
            # Save to cache
            if self.use_cache:
                try:
                    cache_path = self.cache_manager.get_cache_path(chunk_cache_key, "_splits.pkl")
                    with open(cache_path, 'wb') as f:
                        pickle.dump((train, val), f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    self.cache_manager.register_cache(chunk_cache_key, self.file_path, [str(cache_path)])
                    
                except Exception as e:
                    print(f"Failed to save chunk {chunk_idx + 1} to cache: {e}")
            
            # Create datasets
            train_dataset = DKTDataset(train, max_seq=self.max_seq)
            val_dataset = DKTDataset(val, max_seq=self.max_seq)
            
            # Create loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            yield train_loader, val_loader, chunk_idx
            
            # Cleanup
            del chunk, group, train, val, train_dataset, val_dataset
            del train_loader, val_loader
            gc.collect()


# Utility functions để monitor memory
def print_memory_usage():
    """In thông tin memory usage"""
    try:
        import psutil
        import torch
        
        # RAM usage
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB ({ram.percent:.1f}%)")
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_memory:.1f}/{gpu_total:.1f}GB ({gpu_memory/gpu_total*100:.1f}%)")
            
    except ImportError:
        pass


# Cache management utilities
def clear_all_cache(cache_dir="./cache"):
    """Xóa toàn bộ cache"""
    cache_manager = CacheManager(cache_dir)
    cache_manager.clear_cache()


def clear_cache_by_key(cache_key, cache_dir="./cache"):
    """Xóa cache theo key cụ thể"""
    cache_manager = CacheManager(cache_dir)
    cache_manager.clear_cache(cache_key)


def list_cache_info(cache_dir="./cache"):
    """Liệt kê thông tin cache"""
    cache_manager = CacheManager(cache_dir)
    metadata = cache_manager.get_cache_metadata()
    
    print("Cache Information:")
    print("-" * 50)
    
    for key, info in metadata.items():
        created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['created_at']))
        cache_files = info.get('cache_files', [])
        total_size = sum(os.path.getsize(f) for f in cache_files if os.path.exists(f))
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"Key: {key}")
        print(f"  Created: {created_time}")
        print(f"  Source: {info['source_file']}")
        print(f"  Size: {total_size_mb:.1f} MB")
        print(f"  Files: {len(cache_files)}")
        print()


def get_cache_size(cache_dir="./cache"):
    """Tính tổng kích thước cache"""
    cache_manager = CacheManager(cache_dir)
    metadata = cache_manager.get_cache_metadata()
    
    total_size = 0
    for info in metadata.values():
        cache_files = info.get('cache_files', [])
        total_size += sum(os.path.getsize(f) for f in cache_files if os.path.exists(f))
    
    return total_size / (1024 * 1024)  # Return in MB