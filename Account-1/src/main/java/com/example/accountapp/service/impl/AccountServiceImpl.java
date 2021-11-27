package com.example.accountapp.service.impl;

import java.util.List;
import java.util.Optional;

import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import com.example.accountapp.exception.AccountNotFoundException;
import com.example.accountapp.model.Account;
import com.example.accountapp.repository.AccountRepository;
import com.example.accountapp.service.AccountService;

@Service
public class AccountServiceImpl  implements AccountService {

	private AccountRepository accountRepository;
	
	public AccountServiceImpl(AccountRepository accountRepository) {
		super();
		this.accountRepository = accountRepository;
	}

	@Override
	public Account saveAccount(Account account) {
		return accountRepository.save(account);
	}

	@Override
	public List<Account> getAllAccount() {
		return accountRepository.findAll();
	}
		
	@Override
	public List<Account> getAccountByholdingId() {
		//repository.findAll(Sort.by(Sort.Direction.ASC, "seatNumber"));
		return accountRepository.findAll(Sort.by(Sort.Direction.ASC,"holding.hid"));
	}
	
	@Override
	public List<Account> getAccountByholdingname() {
		return accountRepository.findAll(Sort.by(Sort.Direction.DESC,"holding.holdingName"));
	}


	@Override
	public Account updateAccount(Account account, long id) {
		
		Account existingaccount = accountRepository.findById(id).orElseThrow(
			() ->new AccountNotFoundException("Account", "Id", id)	);
		
		existingaccount.setAccountName(account.getAccountName());
		existingaccount.setProfileId(account.getProfileId());
		existingaccount.setTotalBalance(account.getTotalBalance());
		
		accountRepository.save(existingaccount);
		
		return existingaccount;
	}

	@Override
	public Account getAccountById(long id) {
		return accountRepository.findById(id).orElseThrow(
				() ->new AccountNotFoundException("Account", "Id", id));
	}

	@Override
	public void deleteAccount(long id) {
		
		Account existingaccount = accountRepository.findById(id).orElseThrow(
				() ->new AccountNotFoundException("Account", "Id", id)	);
		
		accountRepository.deleteById(id);
		
	}

	


	

}
