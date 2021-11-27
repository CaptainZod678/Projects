package com.example.accountapp.service;

import java.util.List;

import com.example.accountapp.model.Account;

public interface AccountService {
	Account saveAccount(Account account);
	List<Account> getAllAccount();
	List<Account> getAccountByholdingId();
	List<Account> getAccountByholdingname();
	Account getAccountById(long id);
	Account updateAccount(Account account, long id);
	void deleteAccount(long id);

}
